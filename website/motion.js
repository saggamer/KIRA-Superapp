(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const root = document.documentElement;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const motionButton = document.getElementById('motion-toggle');
  const finale = document.getElementById('connect');
  const orbitValue = document.getElementById('orbit-value');
  let paused = reduceMotion.matches;
  let dirty = true, phase = 0, progress = 0, frameId = 0, elapsed = 0, previous = 0, lastDraw = 0;
  const energy = .48;
  let pulseStart = -100;
  const pointer = {x:0,y:0,targetX:0,targetY:0};
  const particleCanvas = document.getElementById('starfield');
  const particles = particleCanvas.getContext('2d');
  const stars = Array.from({length:80},(_,index) => {
    const seed = value => {const n = Math.sin(value*127.1)*43758.54;return n-Math.floor(n);};
    return {x:seed(index+1),y:seed(index+101),depth:.2+seed(index+201),size:.3+seed(index+301)};
  });
  const capabilities = {
    chat:{label:'A NATURAL PLACE TO THINK',title:'Type it. Say it.\nWork it through.',description:'Explore ideas in text or talk with KIRA Live 1. Keep a conversation going, change direction, and make space for the next thought.',model:'KIRA Live 1',prompt:'Can we think through this idea together?',reply:'Start wherever you like.',response:'Talk through your idea, ask a follow-up, or switch to text. Your conversation stays connected in KIRA.',steps:['Text chat','Native voice','Interrupt & continue']},
    research:{label:'CURIOSITY, WITH A PATH FORWARD',title:'Go beyond\nthe first answer.',description:'Explore the web, compare sources, and turn scattered information into a useful brief. Ask a follow-up and keep digging.',model:'Research + tools',prompt:'Research local AI for my next project.',reply:'A question becomes a research plan.',response:'KIRA can gather sources, bring together the useful findings, and prepare a brief with links you can follow.',steps:['Explore sources','Compare findings','Build a brief']},
    create:{label:'IDEAS DESERVE A FORM',title:'Make something\nworth keeping.',description:'Shape rough notes into documents, presentations, spreadsheets, and PDFs. Move from a conversation to an artifact you can use.',model:'Artifact workflows',prompt:'Turn these notes into a presentation.',reply:'Let’s give the story a structure.',response:'Outline the key points, prepare the slides, and review the result. KIRA’s artifact pathways keep the output connected to your task.',steps:['Shape the story','Create an artifact','Review & export']},
    code:{label:'YOUR PROJECT, IN CONTEXT',title:'Find the next\nuseful change.',description:'Explore a project, work through a problem, and prepare code changes with the context of your files and the boundaries of your runtime.',model:'Projects + tools',prompt:'Help me understand this project.',reply:'Start with the pieces that matter.',response:'Read the project context, explain how it fits together, and plan a focused change. File changes remain subject to runtime permissions.',steps:['Read context','Plan a change','Verify the work']},
    tools:{label:'FROM THINKING TO DOING',title:'Bring your tools\ninto the conversation.',description:'Connect plugins and MCP tools, use permission-gated computer actions, and schedule work. KIRA’s runtime validates the steps a model proposes.',model:'Orchestrator V1',prompt:'Help me move this workflow forward.',reply:'A plan that respects your boundaries.',response:'Choose an available tool pathway, check permission where needed, and inspect the result before claiming the task is complete.',steps:['Plugins & MCP','Permission checks','Result evidence']},
    memory:{label:'KEEP THE THREAD OF YOUR WORK',title:'Build on\nwhat came before.',description:'Local conversation history, retrieved context, and workflow state help the next turn stay connected to the work you have already done.',model:'Local context',prompt:'Let’s continue the idea from earlier.',reply:'Pick up the useful context.',response:'KIRA retrieves relevant conversation context and carries workflow state forward within a bounded prompt, so earlier work can inform your next step.',steps:['Local history','Retrieved context','Workflow state']}
  };
  function bindTabs(tabs, select) {
    tabs.forEach((tab,index) => {
      tab.addEventListener('click',()=>select(tab));
      tab.addEventListener('keydown',event => {
        let next;
        if(['ArrowRight','ArrowDown'].includes(event.key))next=(index+1)%tabs.length;
        if(['ArrowLeft','ArrowUp'].includes(event.key))next=(index+tabs.length-1)%tabs.length;
        if(event.key==='Home')next=0;
        if(event.key==='End')next=tabs.length-1;
        if(next!==undefined){event.preventDefault();select(tabs[next]);tabs[next].focus();}
      });
    });
  }
  function setSelected(tabs,chosen,panel) {
    tabs.forEach(tab=>{tab.setAttribute('aria-selected',String(tab===chosen));tab.tabIndex=tab===chosen?0:-1;});
    panel.setAttribute('aria-labelledby',chosen.id);
  }
  const capabilityTabs=[...document.querySelectorAll('[data-capability]')];
  const capabilityPanel=document.getElementById('capability-panel');
  bindTabs(capabilityTabs,tab=>{
    const data=capabilities[tab.dataset.capability];
    setSelected(capabilityTabs,tab,capabilityPanel);
    for(const key of ['label','description','model','prompt','reply','response'])document.getElementById(`cap-${key}`).textContent=data[key];
    const title=document.getElementById('cap-title');
    title.replaceChildren(...data.title.split('\n').flatMap((line,index)=>index?[document.createElement('br'),document.createTextNode(line)]:[document.createTextNode(line)]));
    document.getElementById('cap-steps').replaceChildren(...data.steps.map(label=>{const span=document.createElement('span');span.textContent=label;return span;}));
    capabilityPanel.classList.remove('panel-changing');void capabilityPanel.offsetWidth;capabilityPanel.classList.add('panel-changing');
  });

  const scenarios={
    research:{request:'KIRA, help me research local AI and turn it into a short brief.',plan:'I’d start with a few sources, compare the useful findings, and organize a brief with links. You can interrupt me or change the focus.',permission:'Example permission: save a research brief in your chosen folder?',result:'Example complete: a research brief with an overview, comparison notes, and source links. A real workflow checks the saved file before reporting success.'},
    create:{request:'Turn my product notes into a presentation I can keep working on.',plan:'I’d organize your notes into a story, outline the slides, and prepare an editable presentation. Tell me if you want to change the direction.',permission:'Example permission: create a presentation file from your notes?',result:'Example complete: an editable deck with a clear narrative and a next-steps slide. A real workflow verifies the exported artifact before calling it ready.'},
    code:{request:'Help me understand this project and plan the next change.',plan:'I’d read the relevant project files, explain the structure, and propose a focused change before editing anything.',permission:'Example permission: apply the proposed edit to a project file?',result:'Example complete: the proposed change and a verification summary. A real workflow reviews the actual diff and checks before reporting completion.'}
  };
  const demoTabs=[...document.querySelectorAll('[data-demo]')];
  const demoPanel=document.getElementById('demo-panel');
  const transcript=document.getElementById('transcript');
  const status=document.getElementById('demo-status');
  const play=document.getElementById('demo-play');
  const interrupt=document.getElementById('demo-interrupt');
  const approve=document.getElementById('demo-approve');
  const ledger=[...document.querySelectorAll('.ledger-step')];
  let scenario='research',stage=0,demoState='ready',demoTimer;
  const playLabel=label=>{play.textContent=label;};
  function markLedger(active){ledger.forEach((item,index)=>{item.classList.toggle('active',index===active);item.classList.toggle('done',index<active);});}
  function addMessage(kind,label,message){
    const article=document.createElement('div');article.className=`transcript-message ${kind}`;
    const name=document.createElement('span');name.textContent=label;
    const paragraph=document.createElement('p');paragraph.textContent=message;
    article.append(name,paragraph);transcript.append(article);transcript.scrollTop=transcript.scrollHeight;
  }
  function resetDemo(){
    clearTimeout(demoTimer);stage=0;demoState='ready';status.textContent='Ready for an idea';
    transcript.replaceChildren();const empty=document.createElement('p');empty.className='demo-empty';
    empty.append('A conversation. A plan. A next step.',document.createElement('br'));
    const note=document.createElement('span');note.textContent='Play an example to explore the flow.';empty.append(note);transcript.append(empty);
    approve.hidden=true;interrupt.disabled=true;play.disabled=false;playLabel('Play example ▷');markLedger(-1);
    demoPanel.dataset.state='ready';
  }
  function advanceDemo(){
    const data=scenarios[scenario];
    if(stage===0){transcript.replaceChildren();addMessage('user','YOU / EXAMPLE',data.request);markLedger(0);status.textContent='Understanding the idea';stage=1;}
    else if(stage===1){addMessage('agent','KIRA / EXAMPLE PLAN',data.plan);markLedger(1);status.textContent='Choosing a tool pathway';stage=2;}
    else if(stage===2){addMessage('permission','YOUR DECISION',data.permission);markLedger(2);status.textContent='Waiting for your example approval';demoState='approval';approve.hidden=false;play.disabled=true;playLabel('Waiting for approval');interrupt.disabled=true;demoPanel.dataset.state='approval';return;}
    else if(stage===3){addMessage('agent','KIRA / EXAMPLE ACTION','Approved. This preview now moves to the result-checking step.');markLedger(3);status.textContent='Checking the example result';stage=4;}
    else {addMessage('result','ILLUSTRATIVE RESULT',data.result);markLedger(4);status.textContent='Example workflow complete';demoState='complete';play.disabled=false;playLabel('Replay example ↻');interrupt.disabled=true;demoPanel.dataset.state='complete';return;}
    demoTimer=setTimeout(advanceDemo,stage===4?1100:1400);
  }
  function runDemo(){
    if(demoState==='complete')resetDemo();
    demoState='running';demoPanel.dataset.state='running';play.disabled=true;interrupt.disabled=false;playLabel('Example running…');advanceDemo();
  }
  play.addEventListener('click',runDemo);
  interrupt.addEventListener('click',()=>{
    clearTimeout(demoTimer);demoState='paused';demoPanel.dataset.state='paused';
    status.textContent='Paused — conversation interrupted';play.disabled=false;interrupt.disabled=true;playLabel('Continue example ▷');
    addMessage('agent','INTERRUPTED','The reply pauses here. In KIRA Live 1, your new utterance can take over the conversation. Continue to follow this example.');
  });
  approve.addEventListener('click',()=>{approve.hidden=true;stage=3;runDemo();});
  document.getElementById('demo-reset').addEventListener('click',resetDemo);
  bindTabs(demoTabs,tab=>{scenario=tab.dataset.demo;setSelected(demoTabs,tab,demoPanel);resetDemo();});

  document.getElementById('scene-pulse').addEventListener('click',event=>{
    pulseStart=elapsed;const button=event.currentTarget;button.classList.add('pulsed');dirty=true;wake();
    setTimeout(()=>{button.classList.remove('pulsed');if(paused){pulseStart=-100;dirty=true;wake();}},1000);
  });
  addEventListener('pointermove',event=>{
    if(event.pointerType==='touch'||paused||reduceMotion.matches)return;
    pointer.targetX=(event.clientX/innerWidth-.5)*2;pointer.targetY=(event.clientY/innerHeight-.5)*2;dirty=true;wake();
  },{passive:true});
  root.addEventListener('pointerleave',()=>{pointer.targetX=0;pointer.targetY=0;dirty=true;wake();});
  document.querySelectorAll('.interactive-surface').forEach(surface=>{
    surface.addEventListener('pointermove',event=>{
      if(paused||reduceMotion.matches||event.pointerType==='touch')return;
      const rect=surface.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width,y=(event.clientY-rect.top)/rect.height;
      surface.style.setProperty('--tilt-x',`${(.5-y)*4}deg`);surface.style.setProperty('--tilt-y',`${(x-.5)*4}deg`);
    });
    surface.addEventListener('pointerleave',()=>{surface.style.setProperty('--tilt-x','0deg');surface.style.setProperty('--tilt-y','0deg');});
  });
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.add('visible');observer.unobserve(entry.target);}}),{threshold:.06});
    document.querySelectorAll('.reveal').forEach(element=>observer.observe(element));
  }else document.querySelectorAll('.reveal').forEach(element=>element.classList.add('visible'));
  document.getElementById('year').textContent=String(new Date().getFullYear());
  document.getElementById('replay').addEventListener('click',()=>{window.scrollTo({top:0,behavior:reduceMotion.matches?'instant':'smooth'});document.querySelector('.brand').focus({preventScroll:true});});
  const canvas = document.getElementById('universe');
  const scene = document.querySelector('.universe');
  const artwork = document.getElementById('whitehole-art');
  let artworkReady = false;
  let artTexture;
  let gl;
  try {gl = canvas.getContext('webgl', {alpha: false, antialias: false, depth: false, powerPreference: 'low-power'});} catch (_) {}
  const vertexSource = `attribute vec2 a_position; void main(){ gl_Position=vec4(a_position,0.,1.); }`;
  const fragmentSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform vec2 u_center;
    uniform float u_time;
    uniform float u_size;
    uniform float u_visibility;
    uniform float u_collapse;
    uniform vec2 u_pointer;
    uniform float u_energy;
    uniform float u_pulse;
    uniform vec3 u_tint;
    uniform float u_scroll;
    uniform sampler2D u_art;
    uniform float u_art_ready;
    uniform vec2 u_art_scale;
    float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
    float fbm(vec2 p){float n=0.;float a=.5;for(int i=0;i<4;i++){n+=a*noise(p);p=p*2.03+vec2(7.3,3.2);a*=.5;}return n;}
    mat2 rot(float a){float c=cos(a),s=sin(a);return mat2(c,-s,s,c);}
    void main(){
      vec2 uv=gl_FragCoord.xy/u_resolution;
      vec2 p=(gl_FragCoord.xy-u_center*u_resolution)/u_resolution.y;
      vec3 color=vec3(.027,.030,.043);
      vec2 st=(gl_FragCoord.xy+u_pointer*12.+vec2(0.,u_scroll*22.))/3.;vec2 cell=floor(st);vec2 local=fract(st)-.5;
      float seed=hash(cell);
      float star=smoothstep(.997,1.,seed)*exp(-length(local)*8.);
      star*=.35+.25*sin(u_time*.5+seed*100.);
      color+=star*vec3(.65,.72,1.);
      float mist=fbm(p*3.+vec2(u_time*.004,0.));
      color+=vec3(.019,.025,.055)*mist*exp(-length(p)*1.4);
      if(u_size>.002){
        vec2 q=p/u_size;
        float r=length(q);
        float angle=atan(q.y,q.x);
        if(u_art_ready>.5){
          // Animate the generated artwork as a texture: subtle plasma flow and lens drift.
          vec2 artUV=(rot(u_pointer.x*.025+sin(u_time*.045)*.006)*q)/u_art_scale+.5;
          float flow=.0015+u_energy*.002;
          vec2 ripple=vec2(sin(r*12.-u_time*.35+angle*3.),cos(r*9.+u_time*.23+angle*4.));
          artUV+=ripple*flow*(1.-exp(-r*4.));
          float mask=smoothstep(0.,.07,artUV.x)*smoothstep(0.,.07,artUV.y)*smoothstep(0.,.07,1.-artUV.x)*smoothstep(0.,.07,1.-artUV.y);
          vec3 art=texture2D(u_art,artUV).rgb;
          vec3 tint=mix(vec3(1.),u_tint+vec3(.6),.13);
          float shimmer=1.+.035*sin(r*16.-u_time*.3+angle*2.);
          color+=art*tint*mask*u_visibility*(1.3+u_energy*.5)*shimmer;
          color+=u_tint*exp(-r*r*7.)*.1*u_visibility;
        }else{
          // A restrained loading glow, replaced by the original image as soon as it loads.
          color+=vec3(.65,.78,1.)*exp(-r*r*20.)*.6*u_visibility;
        }
        if(u_pulse>=0. && u_pulse<1.){
          float wave=exp(-abs(r-(.15+u_pulse*2.8))*45.)*(1.-u_pulse);
          color+=u_tint*wave*.28*u_visibility;
        }
      }
      color*=1.-.24*pow(length((uv-.5)*1.3),2.);
      color+=vec3((hash(gl_FragCoord.xy)-.5)*.013);
      color=color/(vec3(1.)+color*.4);
      gl_FragColor=vec4(color,1.);
    }`;
  let uniforms;
  if (gl) {
    try {
      const compile = (type, source) => {
        const shader = gl.createShader(type);gl.shaderSource(shader, source);gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
        return shader;
      };
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Scene link failed');
      gl.useProgram(program);
      const buffer = gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
      const attribute=gl.getAttribLocation(program,'a_position');gl.enableVertexAttribArray(attribute);gl.vertexAttribPointer(attribute,2,gl.FLOAT,false,0,0);
      uniforms=Object.fromEntries(['resolution','center','time','size','visibility','collapse','pointer','energy','pulse','tint','scroll','art','art_ready','art_scale'].map(name=>[name,gl.getUniformLocation(program,`u_${name}`)]));
      artTexture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, artTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.uniform1i(uniforms.art, 0);
    } catch (_) {gl = null;}
  }
  function loadArtwork() {
    if (!artwork.naturalWidth) return;
    if (gl) {
      try {
        gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D, artTexture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, artwork);
        artworkReady = true;
      } catch (_) {gl = null;scene.classList.add('fallback');canvas.hidden = true;}
    }
    scene.dataset.artwork = 'loaded';
    dirty = true; wake();
  }
  artwork.addEventListener('load', loadArtwork);
  if (artwork.complete) loadArtwork();
  if (!gl) {scene.classList.add('fallback'); canvas.hidden=true;}
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();gl=null;scene.classList.add('fallback');canvas.hidden=true;dirty=true;wake();});


  const navLinks=[...document.querySelectorAll('.site-header nav a')].filter(link=>link.hash && document.querySelector(link.hash));
  function measure(){
    const height=innerHeight,maxScroll=Math.max(1,root.scrollHeight-height);
    progress=Math.min(1,Math.max(0,scrollY/maxScroll));
    const start=finale.offsetTop-height*.7;
    phase=Math.min(1,Math.max(0,(scrollY-start)/Math.max(1,maxScroll-start-30)));
    root.style.setProperty('--progress',progress.toFixed(4));orbitValue.textContent=String(Math.round((1-phase)*100));scene.dataset.collapse=phase.toFixed(3);
    document.body.classList.toggle('scrolled',scrollY>80);
    let active='';navLinks.forEach(link=>{if(document.querySelector(link.hash).getBoundingClientRect().top<height*.5)active=link.hash;});
    navLinks.forEach(link=>{link.classList.toggle('active',link.hash===active);if(link.hash===active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
    document.querySelector('.capability-tabs').setAttribute('aria-orientation',innerWidth<=700?'horizontal':'vertical');
    dirty=true;wake();
  }
  function resize(){
    const ratio=Math.min(devicePixelRatio||1,1);canvas.width=Math.round(innerWidth*ratio);canvas.height=Math.round(innerHeight*ratio);
    if(gl)gl.viewport(0,0,canvas.width,canvas.height);particleCanvas.width=canvas.width;particleCanvas.height=canvas.height;measure();
  }
  function draw(){
    const mobile=innerWidth<=700,motionAllowed=!paused&&!reduceMotion.matches;
    pointer.x+=(pointer.targetX-pointer.x)*.07;pointer.y+=(pointer.targetY-pointer.y)*.07;
    const px=motionAllowed?pointer.x:0,py=motionAllowed?pointer.y:0;
    const entering=Math.min(1,phase*3),heroFade=Math.min(1,scrollY/innerHeight),collapse=1-phase*phase;
    const sceneSize=(mobile?.22:.29)*(phase>0?collapse:1-heroFade*.2);
    const centerX=(mobile?.54:.80)*(1-entering)+.5*entering+px*.012;
    const centerY=(mobile?.74:.53)*(1-entering)+.76*entering-py*.009;
    const visibility=phase>0?Math.min(1,collapse*3):1-heroFade*.65;
    root.style.setProperty('--art-width',`${sceneSize*innerHeight*(mobile?3:4.3)}px`);
    root.style.setProperty('--art-x',`${centerX*100}%`);root.style.setProperty('--art-y',`${(1-centerY)*100}%`);
    root.style.setProperty('--art-opacity',String(visibility));root.style.setProperty('--art-brightness','1.5');
    root.style.setProperty('--hero-x',`${px*7}px`);root.style.setProperty('--hero-y',`${motionAllowed?-Math.min(scrollY*.09,70)+py*5:0}px`);
    root.style.setProperty('--controls-x',`${px*-20}px`);root.style.setProperty('--controls-y',`${motionAllowed?py*-14-scrollY*.035:0}px`);
    if(gl){
      gl.uniform2f(uniforms.resolution,canvas.width,canvas.height);gl.uniform2f(uniforms.center,centerX,centerY);
      gl.uniform1f(uniforms.time,elapsed);gl.uniform1f(uniforms.size,sceneSize);gl.uniform1f(uniforms.visibility,visibility);gl.uniform1f(uniforms.collapse,phase);
      gl.uniform2f(uniforms.pointer,px,py);gl.uniform1f(uniforms.energy,energy);gl.uniform1f(uniforms.pulse,motionAllowed?(elapsed-pulseStart)*.7:-1);
      gl.uniform3f(uniforms.tint,.66,.78,1);gl.uniform1f(uniforms.scroll,progress);gl.uniform1f(uniforms.art_ready,artworkReady?1:0);
      gl.uniform2f(uniforms.art_scale,mobile?3:4.3,mobile?2:2.87);gl.drawArrays(gl.TRIANGLES,0,6);
    }
    if(particles){
      const width=particleCanvas.width,height=particleCanvas.height;particles.clearRect(0,0,width,height);
      stars.forEach((star,index)=>{
        const x=((star.x*width+px*star.depth*24)%width+width)%width;
        const y=((star.y*height-(motionAllowed?scrollY*star.depth*.06:0)+py*star.depth*18)%height+height)%height;
        particles.fillStyle=`rgba(190,209,255,${(.16+Math.sin(elapsed*.3+index)*.07)*star.depth})`;
        particles.beginPath();particles.arc(x,y,star.size,0,Math.PI*2);particles.fill();
      });
    }
    dirty=false;
  }
  function tick(now){
    frameId=0;if(document.hidden)return;
    if(previous&&!paused)elapsed+=Math.min((now-previous)/1000,.05);previous=now;
    if(dirty||(!paused&&now-lastDraw>40)){draw();lastDraw=now;}if(!paused)frameId=requestAnimationFrame(tick);
  }
  function wake(){if(!frameId&&!document.hidden)frameId=requestAnimationFrame(tick);}
  function updateMotion(){
    document.body.classList.toggle('motion-paused',paused);motionButton.setAttribute('aria-pressed',String(paused));
    motionButton.querySelector('.motion-label').textContent=paused?'Play motion':'Pause motion';motionButton.querySelector('.motion-icon').textContent=paused?'▷':'Ⅱ';
    motionButton.setAttribute('aria-label',paused?'Play background motion':'Pause background motion');
    if(paused)document.querySelectorAll('.interactive-surface').forEach(surface=>{surface.style.setProperty('--tilt-x','0deg');surface.style.setProperty('--tilt-y','0deg');});
    dirty=true;previous=0;wake();
  }
  motionButton.addEventListener('click',()=>{paused=!paused;updateMotion();});
  reduceMotion.addEventListener('change',event=>{paused=event.matches;updateMotion();});
  addEventListener('scroll',measure,{passive:true});addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',()=>{previous=0;if(document.hidden){cancelAnimationFrame(frameId);frameId=0;}else{dirty=true;wake();}});
  resize();updateMotion();
})();
