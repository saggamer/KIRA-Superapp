(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const button = document.getElementById('motion-toggle');
  const layers = [...document.querySelectorAll('[data-parallax], [data-scroll-orbit]')].map(element => ({
    element, travel: Number(element.dataset.parallax || 0), depth: Number(element.dataset.depthX || 0),
    orbit: Number(element.dataset.scrollOrbit || 0), x: 0, y: 0, angle: 0, visible: true
  }));
  let frame = 0, pointerX = 0, pointerY = 0;
  const isPaused = () => reduced.matches || button?.getAttribute('aria-pressed') === 'true';
  const clamp = (value, max) => Math.max(-max, Math.min(max, value));
  function render() {
    frame = 0;
    if (document.hidden) return;
    const paused = isPaused(), mobile = innerWidth <= 700, strength = mobile ? .4 : 1;
    const scenes = layers.filter(layer => layer.visible || paused).map(layer => {
      const rect = layer.element.getBoundingClientRect();
      const position = clamp((innerHeight * .5 - (rect.top - layer.y + rect.height * .5)) / innerHeight, 1);
      return {layer, position};
    });
    let settling = false;
    for (const {layer, position} of scenes) {
      const targetX = paused || mobile ? 0 : pointerX * layer.depth;
      const targetY = paused ? 0 : position * layer.travel * strength + (mobile ? 0 : pointerY * Math.abs(layer.depth) * .5);
      const targetAngle = paused ? 0 : position * layer.orbit * strength + pointerX * layer.orbit * .08;
      const ease = paused ? 1 : .13;
      layer.x += (targetX - layer.x) * ease;
      layer.y += (targetY - layer.y) * ease;
      layer.angle += (targetAngle - layer.angle) * ease;
      if (Math.abs(targetX-layer.x)+Math.abs(targetY-layer.y)+Math.abs(targetAngle-layer.angle) > .15) settling = true;
      layer.element.style.setProperty('--depth-x', `${layer.x.toFixed(2)}px`);
      layer.element.style.setProperty('--depth-y', `${layer.y.toFixed(2)}px`);
      layer.element.style.setProperty('--orbit-turn', `${layer.angle.toFixed(2)}deg`);
    }
    if (settling && !paused) requestRender();
  }
  function requestRender() { if (!frame && !document.hidden) frame = requestAnimationFrame(render); }
  if ('IntersectionObserver' in window) {
    const lookup = new Map(layers.map(layer => [layer.element,layer]));
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => { lookup.get(entry.target).visible = entry.isIntersecting; });
      requestRender();
    }, {rootMargin:'160px'});
    layers.forEach(layer => observer.observe(layer.element));
  }
  addEventListener('scroll',requestRender,{passive:true});
  addEventListener('resize',requestRender,{passive:true});
  addEventListener('pointermove',event => {
    if (event.pointerType === 'touch' || isPaused()) return;
    pointerX = (event.clientX / innerWidth - .5) * 2;
    pointerY = (event.clientY / innerHeight - .5) * 2;
    requestRender();
  },{passive:true});
  document.documentElement.addEventListener('pointerleave',() => {pointerX=0;pointerY=0;requestRender();});
  document.addEventListener('visibilitychange',() => {
    if (document.hidden) {cancelAnimationFrame(frame);frame=0;} else requestRender();
  });
  function syncModelMotion() {
    const paused = isPaused();
    document.body.classList.toggle('motion-paused',paused);
    if (button?.hasAttribute('data-effects-toggle')) {
      button.setAttribute('aria-label',paused?'Play visual motion':'Pause visual motion');
      button.querySelector('span').textContent=paused?'▷':'Ⅱ';
    }
    requestRender();
  }
  button?.addEventListener('click',() => {
    if (button.hasAttribute('data-effects-toggle')) button.setAttribute('aria-pressed',String(button.getAttribute('aria-pressed')!=='true'));
    syncModelMotion();
  });
  reduced.addEventListener('change',syncModelMotion);
  syncModelMotion();
})();
