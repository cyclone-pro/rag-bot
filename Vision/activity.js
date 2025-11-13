// activity.js
let focusLossStart = null;
let cb = null;
let cfgLocal = { minFocusLossMs: 800 };
let pasteCount = 0, focusLossCount = 0, focusLossMsTotal = 0;

export function initActivityHooks(cfg, onEvent) {
  cfgLocal = { ...cfgLocal, ...cfg };
  cb = onEvent;

  const beginLoss = ()=> { if (focusLossStart == null) focusLossStart = Date.now(); };
  const endLoss = ()=> {
    if (focusLossStart == null) return;
    const dur = Date.now() - focusLossStart;
    focusLossStart = null;
    if (dur >= (cfgLocal.minFocusLossMs || 800)) {
      focusLossCount++;
      focusLossMsTotal += dur;
      cb?.({ focusLossMs: dur });
    }
  };

  document.addEventListener("visibilitychange", ()=> {
    if (document.hidden) beginLoss(); else endLoss();
  });
  window.addEventListener("blur", beginLoss);
  window.addEventListener("focus", endLoss);

  document.addEventListener("paste", ()=> {
    pasteCount++;
    cb?.({ paste: true });
  });
}

export function getActivityState() {
  return { pasteCount, focusLossCount, focusLossMsTotal };
}

export function resetActivity() {
  pasteCount = 0;
  focusLossCount = 0;
  focusLossMsTotal = 0;
  focusLossStart = null;
}
