"""Extra pages for search: one per main service plus an areas page.

Everything here is taken from facts already on the site (services, areas,
Checkatrade score, contact details). app.py registers these routes and adds
them to the sitemap.
"""
import html

SITE = "https://reedsplasteranddecor.co.uk"
PHONE = "07880 256562"
TEL = "+447880256562"

AREAS = ["Chichester", "Bognor Regis", "Littlehampton", "Arundel", "Emsworth", "Havant",
         "Waterlooville", "Petersfield", "Midhurst", "Petworth", "Pulborough", "Storrington",
         "Selsey", "West Wittering", "East Wittering", "Bosham", "Southbourne", "Westbourne",
         "Pagham", "Rustington", "Angmering", "Tangmere", "Lavant", "Funtington"]


def head(title, description, path):
    t, d, url = html.escape(title), html.escape(description), SITE + path
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{t}</title>'
            f'<meta name="description" content="{d}">'
            f'<link rel="canonical" href="{url}">'
            f'<meta property="og:title" content="{t}"><meta property="og:description" content="{d}">'
            f'<meta property="og:url" content="{url}"><meta property="og:type" content="website">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">')


def cta():
    return ('<div class="btns" style="margin-top:26px">'
            f'<a class="btn" href="tel:{TEL}">Call {PHONE}</a>'
            '<a class="btn wa" href="https://wa.me/447880256562" target="_blank" rel="noopener">WhatsApp</a></div>')


def breadcrumb(name, path):
    return ('<script type="application/ld+json">{"@context":"https://schema.org","@type":"BreadcrumbList","itemListElement":['
            f'{{"@type":"ListItem","position":1,"name":"Home","item":"{SITE}/"}},'
            f'{{"@type":"ListItem","position":2,"name":"{name}","item":"{SITE}{path}"}}]}}</script>')


def chips(items):
    return '<div class="chips areas">' + "".join(f'<span class="chip">{html.escape(a)}</span>' for a in items) + "</div>"


PLASTERING = (
    '<section class="band"><div class="wrap narrow"><div class="head reveal"><div class="eyebrow">Plastering &middot; Chichester</div>'
    '<div class="rule"></div><h1>Plastering &amp; skimming in Chichester</h1>'
    '<p class="sub">Walls, ceilings and repairs plastered to a smooth, flat finish that is ready to decorate, '
    'from a local plasterer rated 9.9/10 on Checkatrade.</p></div>'
    '<div class="prose reveal">'
    '<h2>What we plaster</h2>'
    '<p>R. Reeds takes on plastering jobs of every size across Chichester and West Sussex: a full re-skim of a room, '
    'a single ceiling, patch repairs after electrical or plumbing work, chimney breasts, and re-skimming over artex '
    'so a dated textured ceiling becomes flat and modern.</p>'
    '<h2>Homes, period properties and businesses</h2>'
    '<p>Recent jobs include stripping and re-skimming tired walls in a beamed period cottage, plastering a chimney '
    'breast after old boarding was removed, and repairs in a working office carried out without disrupting the business. '
    'Domestic and commercial work are both covered.</p>'
    '<h2>Clean, tidy and ready for paint</h2>'
    '<p>Every job is left tidy, with a finish that is ready for decorating. If you also want the room painted, '
    'R. Reeds can take the job right through to the final coat.</p>'
    '<h2>Getting a free quote</h2>'
    '<p>Quotes are free and written. The quickest way is the chat on this page: describe the job and add a few photos, '
    f'and Rob will come back to you with an estimate. You can also call or WhatsApp {PHONE}.</p>'
    '</div>' + cta() +
    '<p style="margin-top:28px">Also see: <a href="/painting-decorating-chichester">painting &amp; decorating</a> &middot; '
    '<a href="/areas">areas we cover</a> &middot; <a href="/gallery">photos of recent work</a></p>'
    '</div></section>')

DECORATING = (
    '<section class="band"><div class="wrap narrow"><div class="head reveal"><div class="eyebrow">Decorating &middot; Chichester</div>'
    '<div class="rule"></div><h1>Painter &amp; decorator in Chichester</h1>'
    '<p class="sub">Interior and exterior painting and decorating with careful preparation, neat cutting-in and a '
    'proper finish, backed by 248 verified Checkatrade reviews.</p></div>'
    '<div class="prose reveal">'
    '<h2>Inside the home</h2>'
    '<p>Rooms, hallways, stairs and landings, ceilings, feature walls, doors, frames and skirting. '
    'Walls are filled, sanded and prepared before any paint goes on, so the finish lasts.</p>'
    '<h2>Outside</h2>'
    '<p>Masonry, render, timber, doors and garden buildings such as sheds, painted with products suited to the weather.</p>'
    '<h2>Commercial decorating</h2>'
    '<p>Offices and business spaces decorated around your opening hours, including out-of-hours work so there is no '
    'disruption to your staff or customers.</p>'
    '<h2>Plastering and decorating in one job</h2>'
    '<p>Because R. Reeds is a plasterer as well as a decorator, damaged walls and ceilings can be repaired and then '
    'painted by the same person, with no waiting between trades.</p>'
    '<h2>Free written quotes</h2>'
    f'<p>Use the chat to describe the job and send photos, or call or WhatsApp {PHONE}.</p>'
    '</div>' + cta() +
    '<p style="margin-top:28px">Also see: <a href="/plastering-chichester">plastering &amp; skimming</a> &middot; '
    '<a href="/areas">areas we cover</a> &middot; <a href="/gallery">photos of recent work</a></p>'
    '</div></section>')

AREAS_BODY = (
    '<section class="band"><div class="wrap narrow"><div class="head reveal"><div class="eyebrow">Areas we cover</div>'
    '<div class="rule"></div><h1>Plastering &amp; decorating across West Sussex</h1>'
    '<p class="sub">Based in Chichester (PO19) and working within about 20 miles, across West Sussex and into east Hampshire.</p></div>'
    '<div class="prose reveal">'
    '<p>R. Reeds plasters, paints and decorates homes and businesses in these towns and villages:</p></div>'
    + chips(AREAS) +
    '<div class="prose reveal" style="margin-top:26px">'
    '<p>Not on the list? If you are within roughly 20 miles of Chichester, the chances are it is covered. '
    'Ask in the chat and you will get an answer straight away.</p>'
    '<h2>Services in all these areas</h2>'
    '<p><a href="/plastering-chichester">Plastering &amp; skimming</a>: walls, ceilings, repairs and re-skims over artex.<br>'
    '<a href="/painting-decorating-chichester">Painting &amp; decorating</a>: interior, exterior and commercial.</p>'
    '</div>' + cta() + '</div></section>')

PAGES = {
    "/plastering-chichester": ("Plastering & Skimming in Chichester | R. Reeds",
                               "Plastering, skimming, ceiling repairs and re-skims over artex in Chichester and West Sussex. Rated 9.9/10 on Checkatrade. Free written quotes.",
                               "Plastering in Chichester", PLASTERING),
    "/painting-decorating-chichester": ("Painter & Decorator in Chichester | R. Reeds",
                                        "Interior, exterior and commercial painting and decorating in Chichester and West Sussex. 248 verified Checkatrade reviews. Free written quotes.",
                                        "Painting & decorating in Chichester", DECORATING),
    "/areas": ("Areas We Cover: Chichester & West Sussex | R. Reeds",
               "R. Reeds plasters and decorates within 20 miles of Chichester: Bognor Regis, Emsworth, Selsey, Arundel, Littlehampton, Havant and more.",
               "Areas we cover", AREAS_BODY),
}


def render(path, base_style, nav, footer, scripts, widget):
    title, desc, crumb, body = PAGES[path]
    return head(title, desc, path) + base_style + breadcrumb(crumb, path) + "</head><body>" + nav + body + footer + scripts + widget + "</body></html>"
