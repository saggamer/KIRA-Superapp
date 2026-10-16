const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('diagonal', {
  action: (name, value) => ipcRenderer.invoke('diagonal:action', name, value),
  onState: callback => {
    const listener = (_event, state) => callback(state);
    ipcRenderer.on('diagonal:state', listener);
    return () => ipcRenderer.removeListener('diagonal:state', listener);
  },
  onFocusAddress: callback => ipcRenderer.on('diagonal:focus-address', callback),
});
