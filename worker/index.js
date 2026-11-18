// Cloudflare Worker for the R. Reeds chat assistant.
// Pages and images are static files in public/. This file only answers /api/*.
// It is a straight port of the chat and lead email code in app.py, with the
// conversation now held in the visitor's browser instead of server memory.
import { SYSTEM_PROMPT, LEAD_SUMMARY_PROMPT } from "./prompts.js";

const MODEL = "openai/gpt-oss-120b";
const MAX_MESSAGES = 60;          // same cap as app.py (it counted the system prompt too)
const MAX_TEXT = 2000;
const MAX_IMAGES = 6;
const MAX_IMAGE_BYTES = 6 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" };

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/;
const PHONE_RE = /(?<!\d)(?:\+44|0)\d[\d\s\-.]{8,11}(?!\d)/g;
const POSTCODE_RE = /\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b/;
const READY_RE = /\[\[?\s*READY\s*\]?\]/gi;
const CLOSING_RE = new RegExp(
  "\\b(no longer interested|not interested|no thanks|no thank you|" +
  "that'?s all|that'?s it|that'?s everything|nothing else|all good|" +
  "that'?s great thank|thanks that'?s|goodbye|bye for now|no more|" +
  "i'?m good|im good)\\b", "i");

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });

// ---------- contact details (same rules as app.py) ----------
const customerText = (conv) => conv.filter((m) => m.role === "user").map((m) => m.content).join(" ");
function findEmail(conv) { const m = customerText(conv).match(EMAIL_RE); return m ? m[0] : null; }
function findPhone(conv) {
  for (const cand of customerText(conv).match(PHONE_RE) || []) {
    let digits = cand.replace(/\D/g, "");
    if (digits.startsWith("00")) continue;
    if (digits.startsWith("44")) digits = "0" + digits.slice(2);
    if (digits.length === 11 && digits.startsWith("0")) return `${digits.slice(0, 5)} ${digits.slice(5)}`;
  }
  return null;
}
function findPostcode(conv) {
  const m = customerText(conv).match(POSTCODE_RE);
  if (!m) return null;
  const raw = m[0].replace(/\s+/g, "").toUpperCase();
  return raw.slice(0, -3) + " " + raw.slice(-3);
}
const hasContact = (conv) => !!(findEmail(conv) || findPhone(conv));

// ---------- input checks ----------
function cleanHistory(raw) {
  if (!Array.isArray(raw)) return null;
  const out = [];
  for (const m of raw.slice(-MAX_MESSAGES)) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") return null;
    const content = m.content.slice(0, MAX_TEXT).trim();
    if (content) out.push({ role: m.role, content });
  }
  return out;
}

function decodeImage(dataUrl, i) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) return null;
  const comma = dataUrl.indexOf(",");
  if (comma < 0) return null;
  const header = dataUrl.slice(5, comma);
  if (!header.includes(";base64")) return null;
  const type = header.split(";")[0].toLowerCase();
  const ext = ALLOWED_IMAGE_TYPES[type];
  if (!ext) return null;
  const b64 = dataUrl.slice(comma + 1);
  if (!/^[A-Za-z0-9+/]+=*$/.test(b64)) return null;
  const bytes = Math.floor((b64.length * 3) / 4);
  if (!bytes || bytes > MAX_IMAGE_BYTES) return null;
  return { filename: `job-photo-${crypto.randomUUID().slice(0, 8)}-${i + 1}.${ext}`, content: b64 };
}

// ---------- Groq ----------
async function groq(env, body) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { authorization: `Bearer ${env.GROQ_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, ...body }),
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) {
    const err = new Error(`Groq ${res.status}: ${(await res.text()).slice(0, 300)}`);
    err.status = res.status;
    err.retryAfter = Number(res.headers.get("retry-after")) || 0;
    throw err;
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content || "";
}

// ---------- lead email (same layout as app.py) ----------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;" }[c]));

function transcript(conv) {
  return conv.map((m) => `${m.role === "user" ? "Customer" : "Assistant"}: ${m.content}`).join("\n\n");
}

async function summarise(env, conv) {
  try {
    return (await groq(env, {
      messages: [{ role: "system", content: LEAD_SUMMARY_PROMPT }, { role: "user", content: transcript(conv) }],
      // gpt-oss thinks before it writes; 250 tokens cut the summary off.
      max_tokens: 1200, temperature: 0.2, reasoning_effort: "low",
    })).trim();
  } catch (e) {
    console.log("Lead summary failed:", e.message);
    return "";
  }
}

function parseSummary(text) {
  const out = {};
  for (const line of (text || "").split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) out[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
  }
  return out;
}

async function leadFields(env, conv) {
  const s = parseSummary(await summarise(env, conv));
  const pick = (...keys) => {
    for (const k of keys) {
      const v = s[k];
      if (v && !["not specified", "not provided", "n/a", "none", "-"].includes(v.toLowerCase())) return v;
    }
    return null;
  };
  return {
    Name: pick("name"),
    Phone: findPhone(conv),
    Email: findEmail(conv),
    Postcode: findPostcode(conv),
    Area: pick("location / area", "location", "area"),
    Job: pick("job / work wanted", "job", "work wanted"),
    Property: pick("property type (domestic or commercial)", "property type", "property"),
    Budget: pick("approx budget (in gbp £; note if it's a total or a per-room / per-m2 rate)", "approx budget", "budget"),
    "Preferred timing": pick("preferred timing", "timing"),
    Urgency: pick("urgency (1-5 where 1=no rush, 5=urgent - infer from what they said)", "urgency"),
    Notes: pick("other notes", "notes"),
  };
}

function row(label, value) {
  if (!value) return "";
  return '<tr>' +
    `<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#8a8a8a;font-size:13px;white-space:nowrap;vertical-align:top;width:130px">${esc(label)}</td>` +
    `<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#1a1a1a;font-size:14px;font-weight:600">${esc(value)}</td>` +
    '</tr>';
}

function transcriptHtml(conv) {
  return conv.map((m) => {
    const [who, color, bg] = m.role === "user" ? ["Customer", "#0a0a0a", "#f5f4f0"] : ["Reeds Assistant", "#9a7d1a", "#ffffff"];
    return '<div style="margin:0 0 12px">' +
      `<div style="font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:${color};font-weight:700;margin-bottom:4px">${who}</div>` +
      `<div style="background:${bg};border:1px solid #ececec;border-radius:10px;padding:11px 14px;font-size:14px;color:#2a2a2a;line-height:1.5">${esc(m.content).replace(/\n/g, "<br>")}</div>` +
      '</div>';
  }).join("");
}

function urgencyScore(v) { const m = v && String(v).match(/[1-5]/); return m ? Number(m[0]) : 0; }

function urgencyBadge(v) {
  const score = urgencyScore(v);
  if (!score) return "";
  const [bg, fg, label] = {
    1: ["#e8f5e9", "#2e7d32", "1 — No rush"],
    2: ["#f1f8e9", "#558b2f", "2 — Low"],
    3: ["#fff8e1", "#f57f17", "3 — Moderate"],
    4: ["#fff3e0", "#e65100", "4 — Fairly urgent"],
    5: ["#ffebee", "#b71c1c", "5 — URGENT — reply ASAP"],
  }[score];
  return '<div style="margin:0 0 20px">' +
    '<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#999;font-weight:700;margin-bottom:6px">Urgency</div>' +
    `<span style="display:inline-block;background:${bg};color:${fg};border:1px solid ${fg};border-radius:999px;padding:5px 14px;font-size:13px;font-weight:700">${label}</span></div>`;
}

const SHELL_TOP = '<!DOCTYPE html><html><body style="margin:0;background:#f0efea;padding:24px;' +
  'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">' +
  '<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.07)">';

function leadHtml(fields, conv, imageCount, missedPhotos) {
  const { Urgency, ...rest } = fields;
  const rows = Object.entries(rest).map(([k, v]) => row(k, v)).join("");
  const photos = imageCount
    ? `<p style="margin:0 0 20px;font-size:14px;color:#1a1a1a">\u{1F4CE} <strong>${imageCount} photo(s)</strong> attached to this email.</p>` : "";
  const missed = missedPhotos
    ? `<p style="margin:0 0 20px;font-size:14px;color:#1a1a1a">The customer sent ${missedPhotos} photo(s) but left the page before they could be attached. Ask them to send them by WhatsApp.</p>` : "";
  return SHELL_TOP +
    '<div style="background:#0a0a0a;padding:24px 28px">' +
    '<div style="color:#D4AF37;font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:700">R. Reeds</div>' +
    '<div style="color:#fff;font-size:21px;font-weight:700;margin-top:5px">New enquiry from your website</div></div>' +
    '<div style="padding:26px 28px">' +
    '<p style="margin:0 0 20px;font-size:14px;color:#666">Here are the details captured by your website assistant:</p>' +
    urgencyBadge(Urgency) + photos + missed +
    `<table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:8px;overflow:hidden;margin-bottom:28px">${rows}</table>` +
    '<div style="font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#999;font-weight:700;margin-bottom:14px">Full conversation</div>' +
    transcriptHtml(conv) +
    '</div>' +
    '<div style="background:#faf9f6;padding:16px 28px;border-top:1px solid #eee;font-size:12px;color:#aaa">Sent automatically by the R. Reeds website assistant. Chichester &middot; West Sussex</div>' +
    '</div></body></html>';
}

async function sendLead(env, conv, images, missedPhotos) {
  const fields = await leadFields(env, conv);
  const text = ["NEW LEAD - R. Reeds", "========================"];
  for (const [k, v] of Object.entries(fields)) if (v) text.push(`${k}: ${v}`);
  if (images.length) text.push(`Photos attached: ${images.length}`);
  if (missedPhotos) text.push(`Photos not attached (customer left the page): ${missedPhotos}`);
  text.push("========================", "", "Full conversation:", "", transcript(conv));

  const score = urgencyScore(fields.Urgency);
  const prefix = score >= 5 ? "🔴 URGENT — " : score >= 4 ? "🟠 " : "";
  const contact = fields.Phone || fields.Email || "no number yet";
  const bits = [fields.Name, fields.Area || fields.Postcode].filter(Boolean);
  const subject = prefix + "New lead - " + (bits.length ? [...bits, contact].join(" · ") : contact);
  return postResend(env, subject, text.join("\n"), leadHtml(fields, conv, images.length, missedPhotos), images, fields.Email);
}

async function sendPhotoFollowUp(env, conv, images) {
  const phone = findPhone(conv) || "Not provided";
  const email = findEmail(conv) || "Not provided";
  const postcode = findPostcode(conv) || "Not provided";
  const text = "ADDITIONAL PHOTO(S) - R. Reeds\nThis relates to a lead you've already been emailed about.\n" +
    `Phone: ${phone}\nEmail: ${email}\nPostcode: ${postcode}\nPhotos attached: ${images.length}\n`;
  const html = SHELL_TOP +
    '<div style="background:#0a0a0a;padding:22px 28px">' +
    '<div style="color:#D4AF37;font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:700">R. Reeds</div>' +
    '<div style="color:#fff;font-size:19px;font-weight:700;margin-top:5px">More photos for an existing lead</div></div>' +
    '<div style="padding:24px 28px">' +
    `<p style="margin:0 0 18px;font-size:14px;color:#666">This relates to a lead you've already been emailed about. <strong>${images.length} new photo(s)</strong> attached below.</p>` +
    `<table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:8px;overflow:hidden">${row("Phone", phone)}${row("Email", email)}${row("Postcode", postcode)}</table>` +
    '</div></div></body></html>';
  return postResend(env, `Photo added - lead: ${phone}`, text, html, images, null);
}

async function postResend(env, subject, text, html, attachments, replyTo) {
  const payload = { from: env.RESEND_FROM, to: [env.NOTIFY_TO], subject, text, html };
  if (replyTo) payload.reply_to = replyTo;
  if (attachments.length) payload.attachments = attachments;
  if (env.DRY_RUN === "1" || !env.RESEND_API_KEY) {
    console.log("DRY RUN email to", env.NOTIFY_TO, "|", subject, "|", attachments.length, "photo(s)");
    return { dryRun: true, subject, to: env.NOTIFY_TO, text };
  }
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { authorization: `Bearer ${env.RESEND_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Resend ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return { sent: true };
}

// ---------- routes ----------
async function limited(limiter, request) {
  if (!limiter) return false;
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const { success } = await limiter.limit({ key: ip });
  return !success;
}

async function chat(request, env) {
  if (await limited(env.CHAT_LIMIT, request)) {
    return json({ reply: "You're sending messages very quickly - give it a few seconds and try again." });
  }
  const body = await request.json().catch(() => null);
  if (!body) return json({ reply: "Sorry, I didn't catch that - could you type that again?" });
  if ((body.website || "").trim()) return json({ reply: "Thanks!" });

  const conv = cleanHistory(body.history);
  if (!conv || conv.at(-1)?.role !== "user") return json({ reply: "Sorry, I didn't catch that - could you type that again?" });
  if (conv.length >= MAX_MESSAGES - 1) {
    return json({ reply: "Thanks for all the detail! Drop your name and number and R. Reeds will pick this up with you personally." });
  }

  let reply;
  try {
    reply = await groq(env, { messages: [{ role: "system", content: SYSTEM_PROMPT }, ...conv], max_tokens: 256 });
  } catch (e) {
    console.log("Chat completion failed:", e.message);
    if (e.status === 429) return json({ error: "busy", retryAfter: Math.min(Math.max(e.retryAfter, 3), 20) }, 429);
    return json({ error: "failed", reply: "Sorry, I had a brief hiccup there - could you send that again?" }, 503);
  }

  const ready = READY_RE.test(reply);
  READY_RE.lastIndex = 0;
  reply = reply.replace(READY_RE, "").replace("[LEAD_CAPTURED]", "").trim() ||
    "Thanks - that's everything we need for now. R. Reeds will be in touch shortly to arrange your free estimate.";

  const full = [...conv, { role: "assistant", content: reply }];
  const contact = hasContact(full);
  // Same rule as app.py: send straight away when the chat is finished, otherwise
  // the browser sends it after 90 seconds of quiet.
  const lead = !contact ? null
    : (ready || CLOSING_RE.test(conv.at(-1).content) || full.length + 1 >= 24) ? "now" : "later";
  return json({ reply, lead });
}

async function lead(request, env) {
  if (await limited(env.LEAD_LIMIT, request)) return json({ error: "slow down" }, 429);
  const body = await request.json().catch(() => null);
  if (!body || (body.website || "").trim()) return json({ ok: true });
  const conv = cleanHistory(body.history);
  if (!conv || !hasContact(conv)) return json({ error: "no contact details" }, 400);

  const raw = Array.isArray(body.images) ? body.images.slice(0, MAX_IMAGES) : [];
  const images = raw.map(decodeImage).filter(Boolean);
  const missed = Math.max(0, Math.min(MAX_IMAGES, Number(body.missedPhotos) || 0));
  try {
    const result = body.followUp
      ? (images.length ? await sendPhotoFollowUp(env, conv, images) : { skipped: true })
      : await sendLead(env, conv, images, missed);
    return json({ ok: true, ...(env.DRY_RUN === "1" ? { result } : {}) });
  } catch (e) {
    console.log("Lead email failed:", e.message);
    return json({ error: "email failed" }, 502);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
      // One address only: send www.<domain> to <domain>, as the Render version did.
      if (url.hostname.startsWith("www.")) {
        url.hostname = url.hostname.slice(4);
        return Response.redirect(url.toString(), 301);
      }
    if (url.pathname.startsWith("/api/")) {
      if (request.method !== "POST") return json({ error: "method" }, 405);
      const origin = request.headers.get("origin");
      if (origin && new URL(origin).host !== url.host) return json({ error: "origin" }, 403);
      if (url.pathname === "/api/chat") return chat(request, env);
      if (url.pathname === "/api/lead") return lead(request, env);
      return json({ error: "not found" }, 404);
    }
    return env.ASSETS.fetch(request);
  },
};
