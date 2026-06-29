from flask import Flask, request, jsonify, render_template_string, session, Response
import os
import re
import uuid
import html
import base64
import threading
import time
import requests
from groq import Groq

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this-later")
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
# Photos are resized in the browser before upload, so payloads are small.
# This is a safety cap to reject anything abnormally large.
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

_groq_client = None
all_conversations = {}
session_images = {}
notified_sessions = set()
chat_activity = {}


def _decode_image_data_url(data_url):
    """Validate and decode a browser data URL for a customer job photo."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None

    content_type = header[len("data:"):].split(";", 1)[0].lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        return None

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return None

    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[content_type]
    return {
        "filename": f"job-photo-{uuid.uuid4().hex[:8]}.{ext}",
        "content_type": content_type,
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def client_chat(**kwargs):
    """Create the Groq client only when chat is actually used."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client.chat.completions.create(**kwargs)

# --- Email notification settings ---
# Render's free tier blocks direct SMTP (the old Gmail approach), so we
# use Resend instead, which sends over normal HTTPS - not blocked.
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
NOTIFY_TO = os.environ.get("NOTIFY_TO", "rcr.plastering@gmx.com")
RESEND_FROM = os.environ.get("RESEND_FROM", "R. Reeds Website <leads@frontdesk.org.uk>")

# --- Photo upload settings ---------------------------------------------------
# Customers can attach photos of the job; these get emailed with the lead.
# Resizing happens in the browser, so what reaches us here is already small.
MAX_IMAGES_PER_SESSION = 6
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 6 * 1024 * 1024  # per image, after base64 decode

# --- Contact-info extraction -------------------------------------------------
# The lead email is triggered purely by detecting a real phone number or email
# in the conversation (server-side), so we never depend on the AI to flag a lead.

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Matches UK mobile/landline numbers: 07xxx, 01xxx, 02xxx, +447xxx etc.
# No capturing groups so findall returns plain strings.
PHONE_RE = re.compile(r"(?<!\d)(?:\+44|0)\d[\d\s\-\.]{8,11}(?!\d)")
# Full UK postcode, e.g. PO5 3AB, SW1A 1AA, M1 1AE (space optional).
POSTCODE_RE = re.compile(r"\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b")


def _customer_text(conversation):
    """All of the customer's own messages joined together."""
    return " ".join(
        m["content"] for m in conversation if m.get("role") == "user"
    )


def find_email(conversation):
    match = EMAIL_RE.search(_customer_text(conversation))
    return match.group(0) if match else None


def find_phone(conversation):
    text = _customer_text(conversation)
    for candidate in PHONE_RE.findall(text):
        # candidate is always a plain string - no capturing groups in PHONE_RE
        digits = re.sub(r"\D", "", candidate)
        # Reject 00-prefixed numbers (international dialling prefix, not a UK number)
        if digits.startswith("00"):
            continue
        if digits.startswith("44"):
            digits = "0" + digits[2:]
        if len(digits) == 11 and digits.startswith("0"):
            # Format as 07xxx xxxxxx (5 + 6)
            return f"{digits[:5]} {digits[5:]}"
    return None


def find_postcode(conversation):
    match = POSTCODE_RE.search(_customer_text(conversation))
    if not match:
        return None
    # Tidy to canonical form: uppercase, single space before the last 3 chars.
    raw = re.sub(r"\s+", "", match.group(0)).upper()
    return raw[:-3] + " " + raw[-3:]


def has_contact_info(conversation):
    """True only if we genuinely have a way to contact this person back."""
    return bool(find_email(conversation) or find_phone(conversation))


# Phrases that signal the customer is wrapping up - used only as a safety net so
# a lead is never lost if the assistant forgets its closing tag.
CLOSING_RE = re.compile(
    r"\b(no longer interested|not interested|no thanks|no thank you|"
    r"that'?s all|that'?s it|that'?s everything|nothing else|all good|"
    r"that'?s great thank|thanks that'?s|goodbye|bye for now|no more|"
    r"i'?m good|im good)\b",
    re.I,
)


def _looks_like_closing(text):
    return bool(CLOSING_RE.search(text or ""))


def _transcript(conversation):
    lines = []
    for msg in conversation:
        if msg["role"] == "user":
            lines.append(f"Customer: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"Assistant: {msg['content']}")
    return "\n\n".join(lines)


# Prompt that turns a raw chat into a tidy, Checkatrade-style lead.
LEAD_SUMMARY_PROMPT = """You are turning a website chat into a clean lead for a
plastering & decorating company owner. Read the conversation and output EXACTLY
these labelled lines and nothing else. Fill each in from what the customer
actually said; write "Not specified" if they didn't say. Keep each line short.

Name:
Job / work wanted:
Property type (domestic or commercial):
Approx budget (in GBP £; note if it's a total or a per-room / per-m2 rate):
Preferred timing:
Urgency (1-5 where 1=no rush, 5=urgent - infer from what they said):
Location / area:
Other notes:"""


def summarise_lead(conversation):
    """Uses the model to extract a tidy, organised lead from the chat."""
    try:
        resp = client_chat(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": LEAD_SUMMARY_PROMPT},
                {"role": "user", "content": _transcript(conversation)},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Lead summary failed: {e}")
        return None


def _post_resend(subject, text, html_body=None, attachments=None):
    """Low-level send via Resend's HTTPS API (Render's free tier blocks SMTP).

    Sends a plain-text part plus an optional HTML part. `attachments` is a list
    of dicts like {"filename": ..., "b64": <base64>}.
    """
    if not RESEND_API_KEY:
        print("RESEND_API_KEY not set, skipping email")
        return

    payload = {
        "from": RESEND_FROM,
        "to": [NOTIFY_TO],
        "subject": subject,
        "text": text,
    }
    if html_body:
        payload["html"] = html_body
    if attachments:
        # Resend expects: [{"filename": ..., "content": <base64 string>}]
        payload["attachments"] = [
            {"filename": a["filename"], "content": a["b64"]} for a in attachments
        ]

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json=payload,
            timeout=15,
        )
        if response.status_code >= 300:
            print(f"Resend error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def _parse_summary(structured):
    """Turn the model's labelled summary lines into a dict keyed by lowercase label."""
    out = {}
    if not structured:
        return out
    for line in structured.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            out[key.strip().lower()] = val.strip()
    return out


def _lead_fields(conversation):
    """A tidy, ordered set of lead fields - reliable regex first, AI summary for the rest."""
    s = _parse_summary(summarise_lead(conversation))

    def pick(*keys):
        for k in keys:
            v = s.get(k)
            if v and v.lower() not in ("not specified", "not provided", "n/a", "none", "-"):
                return v
        return None

    return {
        "Name": pick("name"),
        "Phone": find_phone(conversation),
        "Email": find_email(conversation),
        "Postcode": find_postcode(conversation),
        "Area": pick("location / area", "location", "area"),
        "Job": pick("job / work wanted", "job", "work wanted"),
        "Property": pick("property type (domestic or commercial)", "property type", "property"),
        "Budget": pick("approx budget", "budget"),
        "Preferred timing": pick("preferred timing", "timing"),
        "Urgency": pick("urgency (1-5 where 1=no rush, 5=urgent - infer from what they said)", "urgency"),
        "Notes": pick("other notes", "notes"),
    }


def _row(label, value):
    if not value:
        return ""
    return (
        '<tr>'
        f'<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#8a8a8a;'
        f'font-size:13px;white-space:nowrap;vertical-align:top;width:130px">{html.escape(label)}</td>'
        f'<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#1a1a1a;'
        f'font-size:14px;font-weight:600">{html.escape(str(value))}</td>'
        '</tr>'
    )


def _transcript_html(conversation):
    rows = []
    for msg in conversation:
        if msg["role"] == "user":
            who, color, bg = "Customer", "#0a0a0a", "#f5f4f0"
        elif msg["role"] == "assistant":
            who, color, bg = "Reeds Assistant", "#9a7d1a", "#ffffff"
        else:
            continue
        text = html.escape(msg["content"]).replace("\n", "<br>")
        rows.append(
            f'<div style="margin:0 0 12px">'
            f'<div style="font-size:11px;letter-spacing:.05em;text-transform:uppercase;'
            f'color:{color};font-weight:700;margin-bottom:4px">{who}</div>'
            f'<div style="background:{bg};border:1px solid #ececec;border-radius:10px;'
            f'padding:11px 14px;font-size:14px;color:#2a2a2a;line-height:1.5">{text}</div>'
            f'</div>'
        )
    return "".join(rows)


def _urgency_badge(urgency_str):
    """Return an HTML urgency badge based on the 1-5 score."""
    if not urgency_str:
        return ""
    # Extract just the digit if present
    m = re.search(r"[1-5]", str(urgency_str))
    if not m:
        return ""
    score = int(m.group(0))
    colours = {
        1: ("#e8f5e9", "#2e7d32", "1 — No rush"),
        2: ("#f1f8e9", "#558b2f", "2 — Low"),
        3: ("#fff8e1", "#f57f17", "3 — Moderate"),
        4: ("#fff3e0", "#e65100", "4 — Fairly urgent"),
        5: ("#ffebee", "#b71c1c", "5 — URGENT — reply ASAP"),
    }
    bg, fg, label = colours.get(score, ("#f5f5f5", "#555", str(score)))
    return (
        f'<div style="margin:0 0 20px">'
        f'<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;'
        f'color:#999;font-weight:700;margin-bottom:6px">Urgency</div>'
        f'<span style="display:inline-block;background:{bg};color:{fg};border:1px solid {fg};'
        f'border-radius:999px;padding:5px 14px;font-size:13px;font-weight:700">'
        f'{label}</span></div>'
    )


def _lead_email_html(fields, conversation, image_count):
    urgency_val = fields.pop("Urgency", None)
    rows = "".join(_row(k, v) for k, v in fields.items())
    photos_line = ""
    if image_count:
        photos_line = (
            '<p style="margin:0 0 20px;font-size:14px;color:#1a1a1a">'
            f'\U0001F4CE <strong>{image_count} photo(s)</strong> attached to this email.</p>'
        )
    urgency_html = _urgency_badge(urgency_val)
    return (
        '<!DOCTYPE html><html><body style="margin:0;background:#f0efea;padding:24px;'
        'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        '<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:14px;'
        'overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.07)">'
        '<div style="background:#0a0a0a;padding:24px 28px">'
        '<div style="color:#D4AF37;font-size:12px;letter-spacing:.18em;text-transform:uppercase;'
        'font-weight:700">R. Reeds</div>'
        '<div style="color:#fff;font-size:21px;font-weight:700;margin-top:5px">'
        'New enquiry from your website</div></div>'
        '<div style="padding:26px 28px">'
        '<p style="margin:0 0 20px;font-size:14px;color:#666">'
        'Here are the details captured by your website assistant:</p>'
        f'{urgency_html}'
        f'{photos_line}'
        '<table style="width:100%;border-collapse:collapse;border:1px solid #eee;'
        f'border-radius:8px;overflow:hidden;margin-bottom:28px">{rows}</table>'
        '<div style="font-size:12px;letter-spacing:.05em;text-transform:uppercase;'
        'color:#999;font-weight:700;margin-bottom:14px">Full conversation</div>'
        f'{_transcript_html(conversation)}'
        '</div>'
        '<div style="background:#faf9f6;padding:16px 28px;border-top:1px solid #eee;'
        'font-size:12px;color:#aaa">Sent automatically by the R. Reeds website assistant. '
        'Chichester &middot; West Sussex</div>'
        '</div></body></html>'
    )


def send_lead_email(conversation, images=None):
    """Emails a tidy, professional lead summary (plus transcript and any photos)."""
    images = images or []
    fields = _lead_fields(conversation)
    transcript = _transcript(conversation)

    # Plain-text fallback for any client that won't render HTML.
    text_lines = ["NEW LEAD - R. Reeds", "========================"]
    for k, v in fields.items():
        if v:
            text_lines.append(f"{k}: {v}")
    if images:
        text_lines.append(f"Photos attached: {len(images)}")
    text_lines += ["========================", "", "Full conversation:", "", transcript]
    text_body = "\n".join(text_lines)

    html_body = _lead_email_html(fields, conversation, len(images))

    # Scannable subject: urgency flag + "New lead - Name · Area · 07..."
    urgency_raw = fields.get("Urgency", "")
    urgency_m = re.search(r"[1-5]", str(urgency_raw)) if urgency_raw else None
    urgency_score = int(urgency_m.group(0)) if urgency_m else 0
    urgent_prefix = "🔴 URGENT — " if urgency_score >= 5 else ("🟠 " if urgency_score >= 4 else "")

    contact = fields.get("Phone") or fields.get("Email") or "no number yet"
    bits = [b for b in (fields.get("Name"), fields.get("Area") or fields.get("Postcode")) if b]
    subject = urgent_prefix + "New lead - " + (" \u00b7 ".join(bits + [contact]) if bits else contact)
    _post_resend(
        subject,
        text_body,
        html_body=html_body,
        attachments=images,
    )


def send_photo_followup(conversation, images):
    """If a photo arrives after the lead email was already sent, forward it on
    so it can't get lost."""
    if not images:
        return

    phone = find_phone(conversation) or "Not provided"
    email = find_email(conversation) or "Not provided"
    postcode = find_postcode(conversation) or "Not provided"

    text_body = (
        "ADDITIONAL PHOTO(S) - R. Reeds\n"
        "This relates to a lead you've already been emailed about.\n"
        f"Phone: {phone}\nEmail: {email}\nPostcode: {postcode}\n"
        f"Photos attached: {len(images)}\n"
    )
    html_body = (
        '<!DOCTYPE html><html><body style="margin:0;background:#f0efea;padding:24px;'
        'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        '<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:14px;overflow:hidden;'
        'box-shadow:0 2px 12px rgba(0,0,0,.07)">'
        '<div style="background:#0a0a0a;padding:22px 28px">'
        '<div style="color:#D4AF37;font-size:12px;letter-spacing:.18em;text-transform:uppercase;'
        'font-weight:700">R. Reeds</div>'
        '<div style="color:#fff;font-size:19px;font-weight:700;margin-top:5px">'
        'More photos for an existing lead</div></div>'
        '<div style="padding:24px 28px">'
        f'<p style="margin:0 0 18px;font-size:14px;color:#666">This relates to a lead you\'ve '
        f'already been emailed about. <strong>{len(images)} new photo(s)</strong> attached below.</p>'
        '<table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:8px;'
        f'overflow:hidden">{_row("Phone", phone)}{_row("Email", email)}{_row("Postcode", postcode)}</table>'
        '</div></div></body></html>'
    )
    _post_resend(f"Photo added - lead: {phone}", text_body, html_body=html_body, attachments=images)


SYSTEM_PROMPT = """
You are the virtual assistant for R. Reeds Plastering & Decorating, a local
plastering, painting and decorating business based in Chichester / West Sussex.
You're the first point of contact for new enquiries.

About the business:
- Services: plastering, skimming, ceiling repairs, plaster repairs, interior
  painting, exterior painting, decorating, woodwork, shed/timber repainting,
  domestic rooms, offices and commercial spaces.
- Local to Chichester, PO19 and surrounding areas.
- Free estimates. Pricing depends on the job, access, prep and finish required.
- Customers can attach photos in the chat with the paperclip.

YOUR TONE:
Write like a friendly local tradesperson sending a quick text. Keep replies
short, helpful and direct. Ask one thing at a time. Do not use long paragraphs,
bullet lists, or customer service phrases like "I'd be happy to assist".

CONVERSATION FLOW - work through these one at a time:
1. Find out what the job is: plastering, skimming, painting, decorating,
   exterior, shed/timber, commercial, etc.
2. Get the scope/size: rooms, walls, ceilings, rough area, repairs, access.
3. Ask if it is domestic or commercial.
4. Ask for photos using the paperclip, or offer a visit if easier.
5. Ask for a rough budget. If they do not want to say, move on.
6. Ask timing/urgency: tomorrow/this week/no rush etc.
7. Ask for their name.
8. Ask for postcode or area.
9. Ask for best phone number or email, then repeat it back to confirm.
10. Only once everything above has been asked, say their enquiry has been sent
    over and R. Reeds will be in touch about a free estimate.

IMPORTANT:
Only add the hidden [[READY]] tag once you have asked about job, scope, domestic
or commercial, photos/visit, budget, urgency, name, area/postcode and confirmed
contact details. The customer never sees [[READY]]. Put it at the very end of the
final wrap-up message, on its own line.
"""




BASE_STYLE = """
<link rel="icon" type="image/jpeg" href="/static/images/logo.jpg">
<meta name="theme-color" content="#050506">
<meta property="og:type" content="website">
<meta property="og:site_name" content="R. Reeds Plastering & Decorating">
<meta property="og:title" content="R. Reeds Plastering & Decorating - Chichester">
<meta property="og:description" content="Plastering, painting and decorating in Chichester and West Sussex. Free quotes, photos welcome.">
<meta property="og:image" content="/static/images/logo.jpg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#f7f4ff; --mut:#bdb4d6; --line:rgba(168,92,255,.22);
    --purple:#a855f7; --pink:#e879f9; --violet:#6d28d9; --bg:#050506; --panel:#0e0d13;
    --paper:#f5f2ee; --paper-ink:#18161c;
  }
  *{box-sizing:border-box} html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:Manrope,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased;overflow-x:hidden}
  a{color:var(--pink)} img,video{max-width:100%;display:block}
  .wrap{max-width:1180px;margin:0 auto;width:100%}.narrow{max-width:820px}
  nav{position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:11px 24px;background:rgba(5,5,6,.82);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
  .brand{display:flex;align-items:center;gap:12px;color:#fff;text-decoration:none;font-weight:800;letter-spacing:.16em;font-size:15px}
  .brand img{width:44px;height:44px;border-radius:10px;object-fit:cover;border:1px solid var(--line)}
  .links{display:flex;align-items:center;gap:22px;flex-wrap:wrap}.links a{color:#ece7ff;text-decoration:none;font-size:13px;font-weight:700;letter-spacing:.02em}.links a:hover{color:var(--pink)}
  .navcta{background:linear-gradient(135deg,var(--violet),var(--pink));padding:10px 16px;border-radius:999px;color:#fff!important;box-shadow:0 8px 24px rgba(168,85,247,.3)}
  .hero{position:relative;overflow:hidden;padding:70px 24px 58px;background:
     radial-gradient(900px 520px at 78% 8%,rgba(124,58,237,.34),transparent 60%),
     radial-gradient(760px 520px at 8% 96%,rgba(232,121,249,.16),transparent 55%),#050506}
  .hero:before{content:"";position:absolute;inset:0;background-image:radial-gradient(rgba(168,85,247,.10) 1px,transparent 1px);background-size:26px 26px;opacity:.5;mask:linear-gradient(#000,transparent 75%)}
  .hero-inner{position:relative;z-index:1;display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:46px;align-items:center}
  .eyebrow{font-size:12px;letter-spacing:.26em;text-transform:uppercase;color:var(--pink);font-weight:800}
  h1{font-family:'Playfair Display',Georgia,serif;font-size:clamp(38px,6vw,70px);line-height:1.0;margin:18px 0 18px;letter-spacing:-.5px}
  h1 .g{background:linear-gradient(120deg,var(--purple),var(--pink));-webkit-background-clip:text;background-clip:text;color:transparent}
  .hero p{font-size:18px;color:#e3daf6;max-width:560px;margin:0 0 26px}
  .btns{display:flex;gap:12px;flex-wrap:wrap}
  .btn{display:inline-flex;align-items:center;gap:9px;justify-content:center;border:0;border-radius:999px;background:linear-gradient(135deg,var(--violet),var(--pink));color:#fff;text-decoration:none;font-weight:800;padding:14px 24px;box-shadow:0 16px 38px rgba(168,85,247,.3);font-size:15px}
  .btn svg{width:18px;height:18px;fill:currentColor}
  .btn.ghost{background:rgba(255,255,255,.05);border:1px solid var(--line);box-shadow:none;color:#fff}
  .btn.wa{background:linear-gradient(135deg,#1faa53,#25d366)}
  .hcard{position:relative}
  .hcard .frame{border-radius:20px;overflow:hidden;border:1px solid var(--line);box-shadow:0 30px 80px rgba(0,0,0,.5);background:#000}
  .hcard .frame img{width:100%;aspect-ratio:4/5;object-fit:cover}
  .hbadge{position:absolute;left:-18px;bottom:26px;background:rgba(12,11,18,.92);border:1px solid var(--line);border-radius:16px;padding:13px 16px;display:flex;align-items:center;gap:11px;box-shadow:0 16px 40px rgba(0,0,0,.5)}
  .hbadge b{font-size:24px;color:#fff;line-height:1;font-family:'Playfair Display',serif}
  .hbadge span{display:block;font-size:11px;color:var(--mut);letter-spacing:.04em}
  .hbadge .st{color:#ffce47;font-size:13px;letter-spacing:1px}
  /* animated marquee */
  .marquee{overflow:hidden;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:rgba(255,255,255,.02)}
  .marquee .track{display:inline-flex;white-space:nowrap;animation:mq 32s linear infinite}
  .marquee:hover .track{animation-play-state:paused}
  .marquee .grp{display:inline-flex;align-items:center;padding:13px 0;font-weight:700;font-size:13.5px;color:#ece6fb;letter-spacing:.02em}
  .marquee .grp i{margin:0 24px;color:var(--purple);font-style:normal}
  .marquee .star{color:#ffce47;margin-right:6px}
  @keyframes mq{from{transform:translateX(0)}to{transform:translateX(-50%)}}
  .band{padding:74px 24px}.paper{background:var(--paper);color:var(--paper-ink)}
  .head{margin-bottom:34px}.head h2{font-family:'Playfair Display',Georgia,serif;font-size:clamp(30px,4.6vw,50px);line-height:1.04;margin:10px 0}.sub{color:#c8bfdc;max-width:700px;font-size:16px}.paper .sub{color:#5f5769}.rule{width:52px;height:3px;background:linear-gradient(90deg,var(--purple),var(--pink));border-radius:999px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:26px;transition:transform .25s,border-color .25s}
  .card:hover{transform:translateY(-4px);border-color:rgba(232,121,249,.4)}
  .card h3{margin:12px 0 8px;font-size:18px;color:#fff}.card p{margin:0;color:var(--mut);font-size:14px}.num{color:var(--pink);font-weight:900;font-size:12px;letter-spacing:.18em}
  .stories{display:grid;gap:26px}
  .story{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(260px,.9fr);gap:30px;align-items:center}
  .story:nth-child(even){grid-template-columns:minmax(260px,.9fr) minmax(0,1.1fr)}.story:nth-child(even) .story-copy{order:-1}
  .ba{position:relative;--pos:50%;border-radius:18px;overflow:hidden;border:1px solid rgba(0,0,0,.12);box-shadow:0 18px 50px rgba(0,0,0,.16);background:#000}
  .ba .spacer img{width:100%;aspect-ratio:4/3;object-fit:cover;opacity:0}
  .ba .layer{position:absolute;inset:0}.ba .layer img{width:100%;height:100%;object-fit:cover}
  .ba .before{clip-path:inset(0 calc(100% - var(--pos)) 0 0)}
  .ba input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;z-index:5}
  .ba:before{content:"";position:absolute;top:0;bottom:0;left:var(--pos);width:2px;background:#fff;z-index:3;box-shadow:0 0 0 1px rgba(0,0,0,.3)}
  .knob{position:absolute;left:var(--pos);top:50%;translate:-50% -50%;z-index:4;width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:#fff;color:#15101b;font-weight:900;box-shadow:0 8px 25px rgba(0,0,0,.3)}.knob:after{content:"\\2039 \\203A";font-size:13px;letter-spacing:1px}
  .tag{position:absolute;top:14px;z-index:4;background:rgba(0,0,0,.72);color:#fff;border:1px solid rgba(255,255,255,.28);border-radius:999px;padding:6px 12px;font-size:11px;font-weight:800;letter-spacing:.12em}.tag.b{left:14px}.tag.a{right:14px}
  .story-copy h3{font-size:26px;margin:0 0 9px}.paper .story-copy h3{color:#1c1622}.story-copy p{color:#5a5264;margin:0}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}.chip{font-size:12px;font-weight:800;border:1px solid rgba(109,40,217,.22);border-radius:999px;padding:7px 12px;color:#52278f;background:#fff}
  .gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
  .shot{margin:0;border-radius:14px;overflow:hidden;background:#000;position:relative;border:1px solid var(--line);cursor:zoom-in}
  .shot img{width:100%;height:100%;aspect-ratio:1/1;object-fit:cover;transition:transform .5s}
  .shot:hover img{transform:scale(1.06)}
  .shot figcaption{position:absolute;left:0;right:0;bottom:0;background:linear-gradient(transparent,rgba(0,0,0,.84));padding:30px 12px 11px;font-size:12px;color:#fff;font-weight:700}
  .reels{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
  .reel{background:#000;border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:0 16px 40px rgba(0,0,0,.16)}
  .reel video{width:100%;aspect-ratio:9/16;object-fit:cover;background:#000}
  .review-summary{display:grid;grid-template-columns:1.05fr .95fr;gap:16px;align-items:stretch;margin-bottom:20px}
  .score-box{border:1px solid rgba(109,40,217,.16);background:#fff;border-radius:20px;padding:26px;display:flex;flex-direction:column;justify-content:center}
  .score-box b{display:block;font-size:48px;line-height:1;color:#2b113d;font-family:'Playfair Display',serif}.score-box .st{color:#ffb400;font-size:18px;letter-spacing:2px;margin:6px 0}.score-box span{display:block;color:#675c72;font-weight:700}
  .review-note{border:1px solid var(--line);background:linear-gradient(135deg,#171022,#0e0a16);color:#f6efff;border-radius:20px;padding:26px}.review-note p{margin:8px 0 0;color:#d8c9ee}
  .reviews-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(258px,1fr));gap:14px}
  .review-card{background:#fff;border:1px solid rgba(109,40,217,.12);border-radius:16px;padding:21px;box-shadow:0 12px 32px rgba(32,14,54,.05)}
  .stars{color:#ffb400;font-weight:900;letter-spacing:.08em;font-size:14px}.review-card h3{margin:9px 0 8px;font-size:16px;line-height:1.3;color:#21162b}.review-card p{margin:0;color:#554d5e;font-size:14px}.review-meta{margin-top:14px;color:#7a7184;font-size:12px;font-weight:800}
  .contact-box{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:28px}.contact-box p{margin:10px 0}
  .prose{color:#ccc3e0;max-width:760px}.prose h3{color:#fff;font-size:18px;margin:26px 0 8px}.prose p{margin:0 0 12px;font-size:15px;line-height:1.7}.prose a{color:var(--pink)}
  .ctaband{background:linear-gradient(135deg,#1a1126,#0c0913);border:1px solid var(--line);border-radius:24px;padding:42px;text-align:center}
  .ctaband h2{font-family:'Playfair Display',serif;font-size:clamp(26px,4vw,40px);margin:0 0 10px;color:#fff}.ctaband p{color:var(--mut);margin:0 0 22px}
  .badges{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:16px}
  .badge{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:800;color:#cdc2e6;border:1px solid var(--line);border-radius:999px;padding:8px 13px;background:rgba(255,255,255,.02)}
  .badge .star{color:#ffce47}
  footer{padding:46px 24px 30px;text-align:center;color:var(--mut);border-top:1px solid var(--line);background:#070708}footer img{width:84px;margin:0 auto 14px;border-radius:12px}
  .wa-float{position:fixed;left:20px;bottom:22px;z-index:999998;width:58px;height:58px;border-radius:50%;background:#25d366;display:grid;place-items:center;box-shadow:0 12px 34px rgba(0,0,0,.4);transition:transform .2s}
  .wa-float:hover{transform:scale(1.07)}.wa-float svg{width:32px;height:32px;fill:#fff}
  .lb{position:fixed;inset:0;z-index:1000000;background:rgba(4,3,8,.93);display:none;align-items:center;justify-content:center;padding:24px;cursor:zoom-out}
  .lb.open{display:flex}.lb img{max-width:92vw;max-height:90vh;border-radius:12px;box-shadow:0 30px 90px rgba(0,0,0,.7)}
  .lb .x{position:absolute;top:18px;right:22px;color:#fff;font-size:34px;font-weight:300;cursor:pointer;line-height:1}
  .reveal{opacity:0;transform:translateY(18px);transition:opacity .7s ease,transform .7s ease}.reveal.in{opacity:1;transform:none}
  @media(max-width:860px){
    .hero{padding:38px 18px 30px}
    .hero-inner{grid-template-columns:1fr;gap:26px}
    .hcard{max-width:320px;margin:0 auto;width:100%}
    .hcard .frame img{aspect-ratio:16/11}
    .hbadge{left:auto;right:10px;bottom:10px;padding:9px 12px}.hbadge b{font-size:19px}
    .story,.story:nth-child(even){grid-template-columns:1fr}.story:nth-child(even) .story-copy{order:0}
    .review-summary{grid-template-columns:1fr}.links a:not(.navcta){display:none}
    .band{padding:50px 18px}.gallery{grid-template-columns:1fr 1fr}
  }
</style>
"""

NAV = """
<nav>
  <a class="brand" href="/"><img src="/static/images/logo.jpg" alt="R. Reeds logo"><span>R.REEDS</span></a>
  <div class="links">
    <a href="/#work">Work</a><a href="/#services">Services</a><a href="/#reviews">Reviews</a>
    <a href="/gallery">Gallery</a><a href="/contact">Contact</a>
    <a class="navcta" href="tel:+447880256562">Free quote</a>
  </div>
</nav>
"""

WA_SVG = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 .4C7.4.4.5 7.3.5 15.9c0 2.8.7 5.4 2 7.8L.3 31.6l8.1-2.1c2.3 1.3 4.9 1.9 7.6 1.9 8.6 0 15.5-6.9 15.5-15.5S24.6.4 16 .4zm0 28.3c-2.4 0-4.7-.6-6.7-1.8l-.5-.3-4.8 1.3 1.3-4.7-.3-.5a12.7 12.7 0 0 1-2-6.8C3.2 8.8 8.9 3.2 16 3.2c7 0 12.7 5.7 12.7 12.7S23 28.7 16 28.7zm7-9.5c-.4-.2-2.3-1.1-2.6-1.3-.3-.1-.6-.2-.8.2-.2.4-.9 1.3-1.1 1.5-.2.2-.4.3-.8.1-.4-.2-1.6-.6-3.1-1.9-1.1-1-1.9-2.3-2.1-2.7-.2-.4 0-.6.2-.8l.6-.7c.2-.2.3-.4.4-.6.1-.2 0-.5 0-.7-.1-.2-.8-2-1.1-2.8-.3-.7-.6-.6-.8-.6h-.7c-.2 0-.6.1-1 .5-.3.4-1.3 1.3-1.3 3.1s1.3 3.6 1.5 3.9c.2.2 2.6 4 6.3 5.6.9.4 1.6.6 2.1.8.9.3 1.7.2 2.3.1.7-.1 2.3-.9 2.6-1.8.3-.9.3-1.6.2-1.8-.1-.1-.3-.2-.7-.4z"/></svg>'

MARQUEE_GRP = ('<span class="grp"><span class="star">&#9733;</span> 9.9/10 on Checkatrade <i>&bull;</i> 244 verified reviews <i>&bull;</i> 20+ years of experience <i>&bull;</i> Which? Trusted Trader <i>&bull;</i> TrustATrader approved <i>&bull;</i> Free written quotes <i>&bull;</i> Plastering <i>&bull;</i> Painting <i>&bull;</i> Decorating <i>&bull;</i> Chichester &amp; West Sussex <i>&bull;</i></span>')
MARQUEE = '<div class="marquee"><div class="track">' + MARQUEE_GRP + MARQUEE_GRP + '</div></div>'

FOOTER = """
<section class="band"><div class="wrap"><div class="ctaband reveal">
  <h2>Got a job in mind?</h2>
  <p>Send a few photos through the chat and get a free, no-obligation estimate.</p>
  <div class="btns" style="justify-content:center"><a class="btn" href="tel:+447880256562">Call 07880 256562</a><a class="btn wa" href="https://wa.me/447880256562" target="_blank" rel="noopener">WhatsApp</a></div>
  <div class="badges"><span class="badge"><span class="star">&#9733;</span> 9.9/10 Checkatrade</span><span class="badge">244 reviews</span><span class="badge">Which? Trusted Trader</span><span class="badge">TrustATrader approved</span><span class="badge">20+ years</span></div>
</div></div></section>
<footer>
  <img src="/static/images/logo.jpg" alt="R. Reeds logo">
  <div style="color:#fff;font-weight:800;letter-spacing:.14em">R.REEDS PLASTERING &amp; DECORATING</div>
  <div style="margin-top:6px">Chichester, PO19 &middot; covering Bognor Regis, Emsworth, Selsey, Arundel &amp; West Sussex</div>
  <div style="margin-top:12px"><a href="tel:+447880256562">07880 256562</a> &nbsp;|&nbsp; <a href="mailto:rcr.plastering@gmx.com">rcr.plastering@gmx.com</a> &nbsp;|&nbsp; <a href="/privacy-policy">Privacy Policy</a></div>
</footer>
<a class="wa-float" href="https://wa.me/447880256562" target="_blank" rel="noopener" aria-label="WhatsApp R. Reeds">""" + WA_SVG + """</a>
<div class="lb" id="lb" onclick="this.classList.remove('open')"><span class="x">&times;</span><img id="lbimg" src="" alt=""></div>
"""

WIDGET_INCLUDE = '<script src="/widget.js"></script>'
GOOGLE_TAG = ""

SCRIPTS = """
<script>
document.querySelectorAll('.ba').forEach(function(ba){
  var range = ba.querySelector('input');
  function update(){ ba.style.setProperty('--pos', range.value + '%'); }
  range.addEventListener('input', update); update();
});
(function(){
  var lb=document.getElementById('lb'), img=document.getElementById('lbimg');
  if(!lb) return;
  document.querySelectorAll('.shot img').forEach(function(im){
    im.addEventListener('click',function(){ img.src=im.src; lb.classList.add('open'); });
  });
})();
(function(){
  var els=document.querySelectorAll('.reveal');
  if(!('IntersectionObserver' in window)){els.forEach(function(e){e.classList.add('in')});return;}
  var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}})},{threshold:.12});
  els.forEach(function(e){io.observe(e)});
})();
</script>
"""

def _ba(before, after, title):
    return ('<div class="ba"><div class="spacer"><img src="/static/images/' + after + '" alt=""></div>'
            '<div class="layer"><img src="/static/images/' + after + '" alt="' + title + ' after"></div>'
            '<div class="layer before"><img src="/static/images/' + before + '" alt="' + title + ' before"></div>'
            '<span class="tag b">Before</span><span class="tag a">After</span><div class="knob"></div>'
            '<input type="range" min="0" max="100" value="50" aria-label="Compare before and after"></div>')

def _shots(items):
    return "".join('<figure class="shot"><img src="/static/images/' + fn + '" alt="' + cap + '" loading="lazy"><figcaption>' + cap + '</figcaption></figure>' for fn, cap in items)

HOME_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>R. Reeds Plastering &amp; Decorating - Chichester</title>
<meta name="description" content="Professional plastering, painting and decorating in Chichester and West Sussex. Free quotes, photo uploads and quick replies.">
<meta name="viewport" content="width=device-width, initial-scale=1">
""" + BASE_STYLE + """</head><body>""" + NAV + """
<header class="hero"><div class="wrap hero-inner">
  <div>
    <div class="eyebrow">Chichester &middot; West Sussex</div>
    <h1>Sharp plastering.<br><span class="g">Flawless</span> decorating.</h1>
    <p>R. Reeds takes on plastering, skimming, painting and decorating for homes and commercial spaces &mdash; clean prep, neat lines, a proper finish every time.</p>
    <div class="btns">
      <a class="btn" href="tel:+447880256562"><svg viewBox="0 0 24 24"><path d="M6.6 10.8a15 15 0 0 0 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.2.4 2.4.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1A17 17 0 0 1 3 4c0-.6.4-1 1-1h3.4c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.4 0 .8-.3 1l-2.1 2.2z"/></svg> Call 07880 256562</a>
      <a class="btn wa" href="https://wa.me/447880256562" target="_blank" rel="noopener">WhatsApp</a>
    </div>
  </div>
  <aside class="hcard reveal">
    <div class="frame"><img src="/static/images/house-2.jpg" alt="White rendered house with purple door, decorated by R. Reeds"></div>
    <div class="hbadge"><div><div class="st">&#9733;&#9733;&#9733;&#9733;&#9733;</div><b>9.9</b></div><div><span style="color:#fff;font-weight:800">Checkatrade</span><span>244 reviews</span></div></div>
  </aside>
</div></header>
""" + MARQUEE + """
<section class="band paper" id="work"><div class="wrap">
  <div class="head reveal"><div class="eyebrow">Real jobs</div><div class="rule"></div><h2>Before &amp; after, dragged to reveal.</h2><p class="sub">Slide across each photo to see the difference &mdash; all real R. Reeds jobs across Chichester and West Sussex.</p></div>
  <div class="stories">
    <article class="story reveal">""" + _ba("cottage-before.jpg","cottage-after.jpg","Cottage room re-plaster") + """<div class="story-copy"><h3>Beamed cottage re-plaster</h3><p>Tired, patchy walls in a period cottage stripped back and re-skimmed to a smooth, even finish ready for paint.</p><div class="chips"><span class="chip">Plastering</span><span class="chip">Period property</span></div></div></article>
    <article class="story reveal">""" + _ba("room-before.jpg","room-after.jpg","Period room repaint") + """<div class="story-copy"><h3>Period room transformed</h3><p>Freshly plastered walls and ceiling taken through to a crisp green finish with bright white woodwork.</p><div class="chips"><span class="chip">Plaster</span><span class="chip">Painting</span><span class="chip">Woodwork</span></div></div></article>
    <article class="story reveal">""" + _ba("landing-before.jpg","landing-after.jpg","Hallway &amp; landing decorate") + """<div class="story-copy"><h3>Hallway &amp; landing</h3><p>Bare plaster brought up to a warm, even grey throughout the stairs and landing, woodwork picked out in white.</p><div class="chips"><span class="chip">Decorating</span><span class="chip">Stairs &amp; landing</span></div></div></article>
    <article class="story reveal">""" + _ba("r.jpg","r7.jpg","Chimney breast plastering") + """<div class="story-copy"><h3>Chimney breast re-plaster</h3><p>Old boarding stripped back, beaded and skimmed to a flat, ready-to-decorate finish.</p><div class="chips"><span class="chip">Plastering</span><span class="chip">Skimming</span></div></div></article>
    <article class="story reveal">""" + _ba("t.jpg","t1.jpg","Commercial office repaint") + """<div class="story-copy"><h3>Commercial office repaint</h3><p>Patch repairs, tidy cutting-in and a calm sage finish across a working office &mdash; out of hours, no disruption.</p><div class="chips"><span class="chip">Commercial</span><span class="chip">Decorating</span></div></div></article>
  </div>
</div></section>

<section class="band" id="services"><div class="wrap">
  <div class="head reveal"><div class="eyebrow">Services</div><div class="rule"></div><h2>The whole job, start to finish.</h2><p class="sub">From repairing damaged walls to the final coat of colour &mdash; and you can send photos straight from the chat while Rob's on site.</p></div>
  <div class="cards">
    <div class="card reveal"><div class="num">01</div><h3>Plastering &amp; skimming</h3><p>Walls, ceilings, chimney breasts, re-skims over artex and patch repairs.</p></div>
    <div class="card reveal"><div class="num">02</div><h3>Painting &amp; decorating</h3><p>Rooms, feature walls, ceilings, doors, frames and skirting &mdash; neat lines guaranteed.</p></div>
    <div class="card reveal"><div class="num">03</div><h3>Exterior &amp; woodwork</h3><p>Masonry, render, timber, doors and outdoor refreshes that hold up to the weather.</p></div>
    <div class="card reveal"><div class="num">04</div><h3>Commercial work</h3><p>Offices and business spaces, worked tidily and around your opening hours.</p></div>
  </div>
</div></section>

<section class="band paper"><div class="wrap">
  <div class="head reveal"><div class="eyebrow">On the tools</div><div class="rule"></div><h2>Straight from site.</h2><p class="sub">A few short clips of recent plastering and decorating work.</p></div>
  <div class="reels reveal">
    <div class="reel"><video controls preload="metadata" playsinline src="/static/videos/reel-1.mp4"></video></div>
    <div class="reel"><video controls preload="metadata" playsinline src="/static/videos/reel-2.mp4"></video></div>
    <div class="reel"><video controls preload="metadata" playsinline src="/static/videos/reel-3.mp4"></video></div>
  </div>
</div></section>

<section class="band"><div class="wrap">
  <div class="head reveal"><div class="eyebrow">Gallery</div><div class="rule"></div><h2>Recent work, real photos.</h2><p class="sub">Tap any photo to view it full size. No stock images &mdash; every shot is a Reeds job.</p></div>
  <div class="gallery reveal">""" + _shots([
      ("house-1.jpg","Exterior &amp; door"),("room-after.jpg","Room repaint"),("cottage-after.jpg","Cottage plaster"),
      ("t1.jpg","Office finish"),("r4.jpg","Chimney skim"),("landing-after.jpg","Landing decorate"),
      ("green-1.jpg","Decorated room"),("kitchen-skim.jpg","Kitchen re-plaster"),
  ]) + """</div>
  <p style="margin-top:22px"><a href="/gallery">Open the full gallery &rarr;</a></p>
</div></section>

<section class="band paper" id="reviews"><div class="wrap">
  <div class="head reveal"><div class="eyebrow">Reviews</div><div class="rule"></div><h2>Rated 9.9/10 on Checkatrade.</h2><p class="sub">From 244 verified reviews across Chichester, Bognor Regis and the surrounding area.</p></div>
  <div class="reveal" style="margin-bottom:20px">
    <div class="score-box" style="flex-direction:row;align-items:center;gap:24px;flex-wrap:wrap"><b style="margin:0">9.9</b><div><div class="st">&#9733;&#9733;&#9733;&#9733;&#9733;</div><span>Checkatrade average from 244 verified reviews</span></div></div>
  </div>
  <div class="reviews-grid">
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Skimming a wall</h3><p>Rob is very polite and a tidy worker. Very pleased with the result.</p><div class="review-meta">Chichester</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Skim over artex</h3><p>Prompt and did a clean job for a reasonable price. Would recommend.</p><div class="review-meta">Richard, Chichester</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Plastering bathroom walls</h3><p>Brilliant. Good price, fast turnaround, polite, extremely happy with the work.</p><div class="review-meta">Steve, Chichester</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Bedroom walls skimmed</h3><p>Professional, friendly and competitively priced. Left the area neat and tidy.</p><div class="review-meta">Sean, Chichester</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Ceiling crack repair</h3><p>Great work and a great price. Quick, clean and helpful advice for future jobs.</p><div class="review-meta">Jacob, Bognor Regis</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Rooms &amp; stairwell plastered</h3><p>Plastered and painted bedrooms plus a hallway and staircase. Very happy, fair prices.</p><div class="review-meta">Catherine, Havant</div></article>
  </div>
</div></section>
""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

GALLERY_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Gallery - R. Reeds</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band"><div class="wrap">
  <div class="head reveal"><div class="eyebrow">Plastering &amp; skimming</div><div class="rule"></div><h2>Plastering work.</h2><p class="sub">Tap any photo to view it full size.</p></div>
  <div class="gallery reveal">""" + _shots([
      ("cottage-after.jpg","Cottage re-plaster"),("r4.jpg","Chimney skim"),("r6.jpg","Fireplace re-plaster"),
      ("r7.jpg","Chimney breast"),("e.jpg","Ceiling skim"),("ee.jpg","Stairwell ceiling"),
      ("kitchen-skim.jpg","Kitchen re-plaster"),("blue-ceiling-1.jpg","Ceiling repair"),
      ("blue-ceiling-2.jpg","Skimmed ceiling"),("room-before.jpg","Room, freshly plastered"),
  ]) + """</div>
  <div class="head reveal" style="margin-top:54px"><div class="eyebrow">Painting &amp; decorating</div><div class="rule"></div><h2>Decorating work.</h2></div>
  <div class="gallery reveal">""" + _shots([
      ("house-1.jpg","Exterior &amp; door"),("house-2.jpg","Rendered front"),("door-purple.jpg","Door painting"),
      ("room-after.jpg","Period room repaint"),("landing-after.jpg","Landing decorate"),("hall-after.jpg","Hallway finish"),
      ("hall-before.jpg","Hallway prep"),("green-1.jpg","Decorated room"),("green-2.jpg","Feature wall"),
      ("green-3.jpg","Finished room"),("t1.jpg","Office, sage finish"),("ttttt.jpg","Office decorated"),
      ("tttt.jpg","Office space"),("ttttttt.jpg","Bright finish"),("ttttttttttt.jpg","Meeting room"),
  ]) + """</div>
</div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

SERVICES_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Services - R. Reeds</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band"><div class="wrap"><div class="head reveal"><div class="eyebrow">Services</div><div class="rule"></div><h2>Plastering, painting &amp; decorating.</h2><p class="sub">Domestic and commercial work across Chichester, Bognor Regis, Emsworth, Selsey, Arundel and nearby West Sussex.</p></div><div class="cards">
<div class="card reveal"><div class="num">01</div><h3>Plastering</h3><p>Skimming, ceilings, patch repairs, chimney breasts, re-skims over artex and damaged walls.</p></div>
<div class="card reveal"><div class="num">02</div><h3>Painting</h3><p>Walls, ceilings, woodwork, feature walls and full room refreshes.</p></div>
<div class="card reveal"><div class="num">03</div><h3>Decorating</h3><p>Preparation, filling, sanding and tidy finishing details throughout.</p></div>
<div class="card reveal"><div class="num">04</div><h3>Exterior &amp; commercial</h3><p>Masonry, render, timber, doors, offices and business spaces worked around your hours.</p></div>
</div></div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

CONTACT_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Contact - R. Reeds</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band"><div class="wrap narrow"><div class="head reveal"><div class="eyebrow">Contact</div><div class="rule"></div><h2>Send a job enquiry.</h2><p class="sub">Use the chat bubble for the fastest quote, or get in touch directly.</p></div><div class="contact-box reveal">
<p><strong style="color:#fff">Phone:</strong> <a href="tel:+447880256562">07880 256562</a></p>
<p><strong style="color:#fff">Email:</strong> <a href="mailto:rcr.plastering@gmx.com">rcr.plastering@gmx.com</a></p>
<p><strong style="color:#fff">WhatsApp:</strong> <a href="https://wa.me/447880256562" target="_blank" rel="noopener">Message us</a></p>
<p><strong style="color:#fff">Area:</strong> Chichester, PO19 and surrounding West Sussex.</p>
</div></div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

PRIVACY_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Privacy Policy - R. Reeds</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band"><div class="wrap narrow">
  <div class="head reveal"><div class="eyebrow">Legal</div><div class="rule"></div><h2>Privacy Policy</h2><p class="sub">How R. Reeds Plastering &amp; Decorating handles the information you share through this website.</p></div>
  <div class="prose reveal">
    <p>This policy explains what information we collect when you use this website or its chat assistant, why we collect it, and how it is kept. By sending an enquiry you agree to the points below.</p>
    <h3>What we collect</h3>
    <p>When you contact us through the chat assistant, contact form, phone, email or WhatsApp we may collect your name, phone number, email address, the details of the job you describe, and any photos you choose to upload. We only collect what you choose to give us &mdash; nothing is gathered without you entering it.</p>
    <h3>Why we collect it</h3>
    <p>We use this information for one purpose: to understand your job, reply to you, and provide a quote or arrange the work. We do not use it for marketing unless you ask us to.</p>
    <h3>How it is handled</h3>
    <p>Enquiry details are sent to our own inbox so we can respond. To run the site we use trusted service providers &mdash; including a chat/AI provider to power the assistant and an email provider to deliver enquiries &mdash; who process the information only to provide that service. We do not sell or rent your information to anyone.</p>
    <h3>How long we keep it</h3>
    <p>We keep enquiry information only for as long as needed to deal with your job and our records, then remove it.</p>
    <h3>Your rights</h3>
    <p>You can ask us what information we hold about you, ask us to correct it, or ask us to delete it at any time. Just get in touch using the details below.</p>
    <h3>Contact</h3>
    <p>R. Reeds Plastering &amp; Decorating, Chichester, West Sussex.<br>Email: <a href="mailto:rcr.plastering@gmx.com">rcr.plastering@gmx.com</a> &nbsp;|&nbsp; Phone: <a href="tel:+447880256562">07880 256562</a></p>
  </div>
</div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

WIDGET_JS = """
(function(){
  var base = new URL(document.currentScript.src).origin;
  var bubble = document.createElement('button');
  bubble.innerHTML = 'Chat';
  bubble.setAttribute('aria-label','Open quote assistant');
  bubble.style.cssText='position:fixed;right:22px;bottom:22px;z-index:999999;border:0;border-radius:999px;background:linear-gradient(135deg,#a855f7,#e879f9);color:white;font-weight:900;padding:15px 18px;box-shadow:0 12px 34px rgba(0,0,0,.34);cursor:pointer';
  var frame = document.createElement('iframe');
  frame.src = base + '/widget-frame';
  function size(){ frame.style.cssText = window.innerWidth <= 640 ? 'position:fixed;inset:0;width:100vw;height:100dvh;border:0;z-index:999999;display:none;background:white' : 'position:fixed;right:22px;bottom:84px;width:410px;height:610px;border:0;border-radius:18px;box-shadow:0 18px 60px rgba(0,0,0,.45);z-index:999999;display:none;background:white'; }
  size(); window.addEventListener('resize',size);
  bubble.onclick=function(){ frame.style.display='block'; bubble.style.display=window.innerWidth<=640?'none':'block'; document.body.style.overflow=window.innerWidth<=640?'hidden':''; };
  window.addEventListener('message',function(e){ if(e.data==='close-au-chat'){ frame.style.display='none'; bubble.style.display='block'; document.body.style.overflow=''; }});
  document.body.appendChild(bubble); document.body.appendChild(frame);
})();
"""

WIDGET_FRAME = """
<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:Manrope,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f2ee;color:#17141b;overflow:hidden}
#chatWindow{height:100dvh;display:flex;flex-direction:column;background:#f5f2ee}#chatHeader{background:#050506;color:white;padding:16px;display:flex;align-items:center;gap:12px;justify-content:space-between}.hbrand{display:flex;gap:10px;align-items:center}.hbrand img{width:42px;height:42px;border-radius:8px;object-fit:cover}.title{font-weight:900}.sub{font-size:12px;color:#d9ccff}.close{font-size:28px;color:#e879f9;cursor:pointer;padding:2px 8px}.progress{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;padding:10px 14px;background:#fff;border-bottom:1px solid #ded6e8}.bar{height:6px;border-radius:99px;background:#e6dff0}.bar.on{background:linear-gradient(90deg,#a855f7,#e879f9)}#status{font-size:12px;color:#6b6178;background:#fff;padding:0 14px 10px;border-bottom:1px solid #ded6e8}#chatbox{flex:1;overflow:auto;padding:16px;-webkit-overflow-scrolling:touch}.msg{max-width:84%;margin:10px 0;padding:12px 14px;border-radius:16px;line-height:1.45;font-size:15px}.bot{background:#fff;border:1px solid #ded6e8}.user{margin-left:auto;background:#100d18;color:white}.photo-msg{padding:5px;background:#100d18}.photo{width:210px;border-radius:12px}#inputRow{flex:none;display:flex;gap:8px;padding:10px;background:white;border-top:1px solid #ded6e8;padding-bottom:max(10px,env(safe-area-inset-bottom))}#userInput{flex:1;min-width:0;border:1px solid #cabfe0;border-radius:999px;padding:12px 14px;font-size:16px;outline:none}#sendBtn,#attachBtn{border:0;border-radius:50%;width:46px;height:46px;display:grid;place-items:center;background:#100d18;color:white;font-weight:900;cursor:pointer;flex:none}#attachBtn{background:#eee8f8;color:#100d18}#fileInput{display:none}.typing{color:#8c819e}
</style></head><body><div id="chatWindow"><div id="chatHeader"><div class="hbrand"><img src="/static/images/logo.jpg"><div><div class="title">R. Reeds Assistant</div><div class="sub">Quote details captured in minutes</div></div></div><div class="close" onclick="window.parent.postMessage('close-au-chat','*')">&times;</div></div><div class="progress"><span class="bar on"></span><span class="bar"></span><span class="bar"></span><span class="bar"></span><span class="bar"></span><span class="bar"></span></div><div id="status">Quote progress: tell us what needs doing</div><div id="chatbox"></div><div id="inputRow"><label id="attachBtn" title="Attach photos"><input type="file" id="fileInput" accept="image/*" multiple onchange="handleFiles(this)">+</label><input type="text" id="hpField" tabindex="-1" autocomplete="off" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0"><input id="userInput" type="text" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()"><button id="sendBtn" onclick="sendMessage()">></button></div></div>
<script>
var messages=0; addMessage("Hi, I can help get a free quote for plastering, painting or decorating. What needs doing?", "bot");
function updateProgress(){var n=Math.min(6,Math.ceil(messages/2));document.querySelectorAll('.bar').forEach(function(b,i){b.classList.toggle('on',i<n)});document.getElementById('status').textContent='Quote progress: '+n+'/6 details captured';}
function addMessage(t,s){var c=document.getElementById('chatbox'),d=document.createElement('div');d.className='msg '+s;d.textContent=t;c.appendChild(d);c.scrollTop=c.scrollHeight;if(s==='user'){messages++;updateProgress();}}
function typing(){var c=document.getElementById('chatbox'),d=document.createElement('div');d.id='typing';d.className='msg bot typing';d.textContent='...';c.appendChild(d);c.scrollTop=c.scrollHeight}
function untyping(){var t=document.getElementById('typing');if(t)t.remove()}
async function sendMessage(){var i=document.getElementById('userInput'),m=i.value.trim();if(!m)return;addMessage(m,'user');i.value='';typing();try{var r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m,website:(document.getElementById('hpField')||{}).value||''}),credentials:'same-origin'});var d=await r.json();untyping();addMessage(d.reply,'bot')}catch(e){untyping();addMessage("Sorry, that did not send. Please try again.",'bot')}}
function addImage(src){var c=document.getElementById('chatbox'),d=document.createElement('div'),img=document.createElement('img');d.className='msg user photo-msg';img.className='photo';img.src=src;d.appendChild(img);c.appendChild(d);c.scrollTop=c.scrollHeight;messages++;updateProgress();}
function resizeImage(file){return new Promise(function(resolve,reject){var reader=new FileReader();reader.onload=function(){var img=new Image();img.onload=function(){var max=1600,w=img.naturalWidth,h=img.naturalHeight;if(Math.max(w,h)>max){if(w>=h){h=Math.round(h*max/w);w=max}else{w=Math.round(w*max/h);h=max}}var canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;var ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.drawImage(img,0,0,w,h);resolve(canvas.toDataURL('image/jpeg',.82))};img.onerror=reject;img.src=reader.result};reader.onerror=reject;reader.readAsDataURL(file)})}
async function handleFiles(input){var files=Array.from(input.files||[]);input.value='';for(const file of files){if(!file.type.startsWith('image/')){addMessage('Please choose a photo file.','bot');continue}try{var dataUrl=await resizeImage(file);addImage(dataUrl);var r=await fetch('/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:dataUrl}),credentials:'same-origin'});var d=await r.json();addMessage(d.reply,'bot')}catch(e){addMessage('Sorry, I could not upload that photo. Try another JPG or PNG.','bot')}}}
</script></body></html>
"""

def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

@app.route("/sitemap.xml")
def sitemap():
    pages = ["/", "/services", "/gallery", "/contact", "/privacy-policy"]
    base = "https://reeds-demo.onrender.com"
    urls = "".join(f"<url><loc>{base}{p}</loc></url>" for p in pages)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'
    return Response(xml, mimetype="application/xml")

@app.route("/robots.txt")
def robots():
    return Response("User-agent: *\nAllow: /\nSitemap: https://reeds-demo.onrender.com/sitemap.xml", mimetype="text/plain")

@app.route("/")
def home():
    ensure_session()
    return render_template_string(HOME_PAGE)

@app.route("/services")
def services():
    ensure_session()
    return render_template_string(SERVICES_PAGE)

@app.route("/gallery")
def gallery():
    ensure_session()
    return render_template_string(GALLERY_PAGE)

@app.route("/contact")
def contact():
    ensure_session()
    return render_template_string(CONTACT_PAGE)

@app.route("/privacy")
@app.route("/privacy-policy")
def privacy():
    ensure_session()
    return render_template_string(PRIVACY_PAGE)

@app.route("/widget.js")
def widget_js():
    return Response(WIDGET_JS, mimetype="application/javascript")

@app.route("/widget-frame")
def widget_frame():
    ensure_session()
    return render_template_string(WIDGET_FRAME)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    if session_id not in all_conversations:
        all_conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation = all_conversations[session_id]

    data = request.get_json(silent=True) or {}

    # Honeypot: a hidden field real visitors never see or fill. If it's populated,
    # it's almost certainly a bot - quietly stop before spending Groq/Resend.
    if (data.get("website") or "").strip():
        return jsonify({"reply": "Thanks!"})

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Sorry, I didn't catch that - could you type that again?"})

    # Per-session rate limiting to protect against abuse running up Groq/Resend
    # cost: max 20 messages a minute, plus a hard cap per visitor.
    now = time.time()
    recent = [t for t in chat_activity.get(session_id, []) if now - t < 60]
    if len(recent) >= 20:
        return jsonify({"reply": "You're sending messages very quickly - give it a few seconds and try again."})
    if len(conversation) >= 60:
        return jsonify({"reply": "Thanks for all the detail! Drop your name and number and R. Reeds will pick this up with you personally."})
    recent.append(now)
    chat_activity[session_id] = recent

    conversation.append({"role": "user", "content": user_message})

    try:
        response = client_chat(
            model="llama-3.3-70b-versatile",
            messages=conversation,
            max_tokens=256,
            timeout=20,
        )
        ai_reply = response.choices[0].message.content
    except Exception as e:
        # Never leave the customer staring at a frozen chat. Drop the message we
        # just appended so they can retry cleanly, and reply with a gentle note.
        print(f"Chat completion failed: {e}")
        conversation.pop()
        return jsonify({
            "reply": "Sorry, I had a brief hiccup there - could you send that again?"
        })

    # Strip any internal signal tags so they can never reach the customer.
    lead_ready = bool(re.search(r"\[\[?\s*READY\s*\]?\]", ai_reply, re.I))
    ai_reply = re.sub(r"\[\[?\s*READY\s*\]?\]", "", ai_reply)
    ai_reply = ai_reply.replace("[LEAD_CAPTURED]", "").strip()
    if not ai_reply:
        ai_reply = ("Thanks - that's everything we need for now. R. Reeds "
                    "will be in touch shortly to arrange your free estimate.")

    conversation.append({"role": "assistant", "content": ai_reply})

    # Only email once the assistant has genuinely finished gathering EVERYTHING.
    # It signals this with the internal [[READY]] tag, which it only adds after
    # working through the whole checklist (job, scope, budget, area, contact...).
    # We deliberately do NOT send on wrap-up phrases or a low turn count, because
    # that was firing before budget/postcode were collected. The fallbacks below
    # are conservative - only if the visitor clearly signs off, or a very long
    # chat - so a lead is never lost, but normal chats wait for the full set of
    # questions. Sent at most once per visitor.
    if session_id not in notified_sessions and has_contact_info(conversation):
        if lead_ready or _looks_like_closing(user_message) or len(conversation) >= 24:
            notified_sessions.add(session_id)
            conversation_copy = list(conversation)
            images_copy = list(session_images.get(session_id, []))
            send_lead_email(conversation_copy, images_copy)

    return jsonify({"reply": ai_reply})


@app.route("/upload", methods=["POST"])
def upload_endpoint():
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    if session_id not in all_conversations:
        all_conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation = all_conversations[session_id]

    data = request.get_json(silent=True) or {}
    image = _decode_image_data_url(data.get("image", ""))
    if image is None:
        return (
            jsonify({"reply": "Sorry, I couldn't read that image. Please try a JPG or PNG photo."}),
            400,
        )

    images = session_images.setdefault(session_id, [])
    if len(images) >= MAX_IMAGES_PER_SESSION:
        return jsonify({
            "reply": "Thanks - that's plenty of photos for now. Leave your name and number and we'll take a look and get you a quote."
        })

    images.append(image)

    # Keep the transcript (and the AI) aware that a photo came in.
    conversation.append({"role": "user", "content": "(Customer attached a photo of the job)"})
    reply = (
        "Thanks, got the photo - that really helps. You can add another, "
        "or tell me if that is all and I will carry on with the quote details."
    )
    conversation.append({"role": "assistant", "content": reply})

    # If we've already emailed this lead, forward the new photo as a follow-up
    # so it doesn't get lost.
    if session_id in notified_sessions:
        send_photo_followup(list(conversation), [image])

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
