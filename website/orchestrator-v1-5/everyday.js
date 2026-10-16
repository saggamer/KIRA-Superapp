(() => {
  'use strict';
  const paths = {
    browser: {label:'BROWSE WITH CONTEXT', title:['Stay with the task.','Read the page first.'], description:'The browser pathway works with Diagonal Browser observations, checks current targets, and carries verified results into the next decision.', tags:['Browser adapter','Fresh observations','Grounded targets'], prompt:'Compare these two pages, note the differences, and give me a brief with source links. Ask before submitting anything.', steps:['Observe the current pages','Compare grounded findings','Return a sourced summary'], boundary:'After an authorized external action, the controller requires a fresh observation.'},
    apps: {label:'CONNECT THE NEXT STEP', title:['Bring your apps','into the workflow.'], description:'The control pathway proposes app, MCP, and supported subagent actions. KIRA’s handlers execute accepted tool blocks and enforce the required permissions.', tags:['Control adapter','MCP & app tools','Runtime permissions'], prompt:'Use the available tools to organize this task. First inspect the context, show me the plan, and ask before making changes.', steps:['Inspect tools and task context','Propose the supported action','Check permission and the result'], boundary:'Available tools depend on your KIRA setup and connected services.'},
    decks: {label:'GIVE AN IDEA A STRUCTURE', title:['From source material','to a presentation.'], description:'The presentation pathway proposes research, image, and slide actions. KIRA’s slide specialist creates the file and verifies the artifact before completion is reported.', tags:['Presentation adapter','Grounded images & links','Verified PPTX'], prompt:'Create a six-slide presentation from these notes. Keep the supplied facts, use source-backed images, include links, and return the verified PPTX path.', steps:['Gather sources and image evidence','Prepare a grounded slide structure','Create and verify the PPTX'], boundary:'Autonomous deck generation remains limited; a repair turn or effort limit may stop a workflow.'}
  };
  const efforts = {
    low: {label:'LOW / A FOCUSED NEXT STEP',title:'Start with a smaller budget.',description:'One proposal per browser decision, with a bounded allowance for short workflows.',proposals:1,rounds:4,tools:4,corrections:1,tokens:'1,536',minutes:2},
    medium: {label:'MEDIUM / ROOM TO CONTINUE',title:'Keep the workflow moving.',description:'One proposal per browser decision, with a larger allowance for steps and corrections than Low.',proposals:1,rounds:6,tools:8,corrections:2,tokens:'3,072',minutes:4},
    high: {label:'HIGH / COMPARE THE OPTIONS',title:'Give the next action another look.',description:'Two proposals per browser decision. The controller compares grounded options before selecting one action.',proposals:2,rounds:8,tools:12,corrections:3,tokens:'6,144',minutes:8},
    diagonal: {label:'DIAGONAL / MORE ROOM TO REVIEW',title:'Explore options. Execute one.',description:'Up to three proposals share the same verified request and browser evidence. The controller rejects invalid proposals and selects one next action.',proposals:3,rounds:12,tools:16,corrections:4,tokens:'8,192',minutes:15}
  };
  function tabs(selector, panelId, update) {
    const buttons=[...document.querySelectorAll(selector)],panel=document.getElementById(panelId);
    function select(button) {
      buttons.forEach(item=>{item.setAttribute('aria-selected',String(item===button));item.tabIndex=item===button?0:-1;});
      panel.setAttribute('aria-labelledby',button.id);update(button);
    }
    buttons.forEach((button,index)=>{
      button.addEventListener('click',()=>select(button));
      button.addEventListener('keydown',event=>{
        let next;
        if(['ArrowRight','ArrowDown'].includes(event.key))next=(index+1)%buttons.length;
        if(['ArrowLeft','ArrowUp'].includes(event.key))next=(index+buttons.length-1)%buttons.length;
        if(event.key==='Home')next=0;
        if(event.key==='End')next=buttons.length-1;
        if(next!==undefined){event.preventDefault();select(buttons[next]);buttons[next].focus();}
      });
    });
  }
  const setText=(id,value)=>{document.getElementById(id).textContent=value;};
  tabs('[data-path]','path-panel',button=>{
    const data=paths[button.dataset.path];
    ['label','description','prompt','boundary'].forEach(key=>setText(`path-${key}`,data[key]));
    const title=document.getElementById('path-title');title.replaceChildren(document.createTextNode(data.title[0]),document.createElement('br'),document.createTextNode(data.title[1]));
    document.getElementById('path-tags').replaceChildren(...data.tags.map(text=>{const span=document.createElement('span');span.textContent=text;return span;}));
    document.getElementById('path-steps').replaceChildren(...data.steps.map((text,index)=>{const span=document.createElement('span'),number=document.createElement('i');number.textContent=`0${index+1}`;span.append(number,document.createTextNode(text));return span;}));
    setText('copy-status','');
  });
  tabs('[data-effort]','effort-panel',button=>{
    const data=efforts[button.dataset.effort];
    ['label','title','description'].forEach(key=>setText(`effort-${key}`,data[key]));
    ['proposals','rounds','tools','corrections'].forEach(key=>setText(`budget-${key}`,data[key]));
    setText('budget-extra',`Up to ${data.tokens} generated proposal tokens and a ${data.minutes}-minute time cap.`);
  });
  let copyRequest=0;
  document.getElementById('copy-example').addEventListener('click',async()=>{
    const request=++copyRequest,text=document.getElementById('path-prompt').textContent;
    try {await navigator.clipboard.writeText(text);if(request===copyRequest)setText('copy-status','Example copied. Paste it into KIRA and add your own context.');}
    catch (_) {if(request===copyRequest)setText('copy-status','Select and copy the example above, then paste it into KIRA.');}
  });
  const reduceMotion=matchMedia('(prefers-reduced-motion: reduce)');
  const art=document.querySelector('.model-art');
  art.addEventListener('pointermove',event=>{
    if(reduceMotion.matches||event.pointerType==='touch')return;
    const rect=art.getBoundingClientRect();art.style.setProperty('--drift-x',`${((event.clientX-rect.left)/rect.width-.5)*12}px`);art.style.setProperty('--drift-y',`${((event.clientY-rect.top)/rect.height-.5)*12}px`);
  });
  art.addEventListener('pointerleave',()=>{art.style.setProperty('--drift-x','0px');art.style.setProperty('--drift-y','0px');});
})();
