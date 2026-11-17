// Chat window logic for the R. Reeds assistant (runs inside /widget-frame).
// Looks and behaves like the old Flask version; the conversation is kept in the
// browser (sessionStorage) and the lead email is sent from here.
var GREETING = "Hi, I can help get a free quote for plastering, painting or decorating. What needs doing?";
var PHOTO_REPLY = "Thanks, got the photo - that really helps. You can add another, or tell me if that is all and I will carry on with the quote details.";
var MAX_PHOTOS = 6;
var FALLBACK_MS = 90 * 1000;
var STORE = "reedsChat";

var state = { history: [], photos: [], sentPhotos: 0, sent: false, hasContact: false, shown: [] };
var messages = 0, timer = 0, busy = false;

function save() {
  try { sessionStorage.setItem(STORE, JSON.stringify(state)); }
  catch (e) { try { var s = Object.assign({}, state, { photos: [] }); sessionStorage.setItem(STORE, JSON.stringify(s)); } catch (e2) {} }
}

function updateProgress(){var n=Math.min(6,Math.ceil(messages/2));document.querySelectorAll('.bar').forEach(function(b,i){b.classList.toggle('on',i<n)});document.getElementById('status').textContent='Quote progress: '+n+'/6 details captured';}
function addMessage(t,s,keep){var c=document.getElementById('chatbox'),d=document.createElement('div');d.className='msg '+s;d.textContent=t;c.appendChild(d);c.scrollTop=c.scrollHeight;if(s==='user'){messages++;updateProgress();}if(keep!==false){state.shown.push({t:t,s:s});save();}}
function typing(){var c=document.getElementById('chatbox'),d=document.createElement('div');d.id='typing';d.className='msg bot typing';d.textContent='...';c.appendChild(d);c.scrollTop=c.scrollHeight}
function untyping(){var t=document.getElementById('typing');if(t)t.remove()}
function addImage(src,keep){var c=document.getElementById('chatbox'),d=document.createElement('div'),img=document.createElement('img');d.className='msg user photo-msg';img.className='photo';img.src=src;d.appendChild(img);c.appendChild(d);c.scrollTop=c.scrollHeight;messages++;updateProgress();if(keep!==false){state.shown.push({img:state.photos.length-1});save();}}

function pushUser(text) {
  var last = state.history[state.history.length - 1];
  if (last && last.role === 'user') last.content += '\n' + text;
  else state.history.push({ role: 'user', content: text });
}

function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

async function post(path, body, opts) {
  for (var i = 0; ; i++) {
    var r = await fetch(path, Object.assign({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, opts || {}));
    var d = await r.json().catch(function () { return {}; });
    if (r.status === 429 && d.error === 'busy' && i < 2) { await wait((d.retryAfter || 10) * 1000); continue; }
    if (!r.ok) throw Object.assign(new Error(d.error || r.status), { data: d });
    return d;
  }
}

function hp() { return (document.getElementById('hpField') || {}).value || ''; }

async function sendLead() {
  clearTimeout(timer);
  if (state.sent || !state.hasContact) return;
  state.sent = true; save();
  try {
    await post('/api/lead', { history: state.history, images: state.photos, website: hp() });
    state.sentPhotos = state.photos.length; save();
  } catch (e) { state.sent = false; save(); }
}

async function sendFollowUp(img) {
  try { await post('/api/lead', { history: state.history, images: [img], followUp: true, website: hp() }); } catch (e) {}
}

async function sendMessage(){
  var i=document.getElementById('userInput'),m=i.value.trim();if(!m||busy)return;
  addMessage(m,'user');i.value='';typing();busy=true;
  pushUser(m); save();
  try{
    var d=await post('/api/chat',{history:state.history,website:hp()});
    untyping();
    if(d.lead!==undefined){
      state.history.push({role:'assistant',content:d.reply});
      if(d.lead){state.hasContact=true;}
      save();
      if(d.lead==='now'){sendLead();}
      else if(d.lead==='later'&&!state.sent){clearTimeout(timer);timer=setTimeout(sendLead,FALLBACK_MS);}
    }else{state.history.pop();save();}
    addMessage(d.reply,'bot');
  }catch(e){
    untyping();
    state.history.pop();save();
    addMessage((e.data&&e.data.reply)||"Sorry, that did not send. Please try again.",'bot');
  }
  busy=false;
}

function resizeImage(file){return new Promise(function(resolve,reject){var reader=new FileReader();reader.onload=function(){var img=new Image();img.onload=function(){var max=1600,w=img.naturalWidth,h=img.naturalHeight;if(Math.max(w,h)>max){if(w>=h){h=Math.round(h*max/w);w=max}else{w=Math.round(w*max/h);h=max}}var canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;var ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.drawImage(img,0,0,w,h);resolve(canvas.toDataURL('image/jpeg',.82))};img.onerror=reject;img.src=reader.result};reader.onerror=reject;reader.readAsDataURL(file)})}

async function handleFiles(input){
  var files=Array.from(input.files||[]);input.value='';
  for(const file of files){
    if(!file.type.startsWith('image/')){addMessage('Please choose a photo file.','bot');continue}
    if(state.photos.length>=MAX_PHOTOS){addMessage("Thanks - that's plenty of photos for now. Leave your name and number and we'll take a look and get you a quote.",'bot');continue}
    try{
      var dataUrl=await resizeImage(file);
      state.photos.push(dataUrl);addImage(dataUrl);
      pushUser('(Customer attached a photo of the job)');
      state.history.push({role:'assistant',content:PHOTO_REPLY});save();
      addMessage(PHOTO_REPLY,'bot');
      if(state.sent){state.sentPhotos=state.photos.length;save();sendFollowUp(dataUrl);}
    }catch(e){addMessage('Sorry, I could not upload that photo. Try another JPG or PNG.','bot')}
  }
}

// If they leave after giving a number but before the email went, send it now.
// Browsers cap what can be sent while a page closes, so photos can't go with it.
window.addEventListener('pagehide', function () {
  if (state.sent || !state.hasContact) return;
  state.sent = true; save();
  try {
    fetch('/api/lead', { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history: state.history, images: [], missedPhotos: state.photos.length, website: hp() }) });
  } catch (e) {}
});

(function restore() {
  var saved = null;
  try { saved = JSON.parse(sessionStorage.getItem(STORE) || 'null'); } catch (e) {}
  if (saved && Array.isArray(saved.history) && saved.shown && saved.shown.length) {
    state = Object.assign(state, saved);
    state.shown.forEach(function (m) {
      if (m.img !== undefined) { if (state.photos[m.img]) addImage(state.photos[m.img], false); }
      else addMessage(m.t, m.s, false);
    });
    if (state.hasContact && !state.sent) timer = setTimeout(sendLead, FALLBACK_MS);
  } else {
    addMessage(GREETING, 'bot');
  }
})();
