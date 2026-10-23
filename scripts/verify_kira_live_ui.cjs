const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require('../diagonal_browser/node_modules/playwright');

(async () => {
    const browser = await chromium.launch({executablePath: process.env.DIAGONAL_CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true});
    try {
        const page = await browser.newPage({viewport: {width: 1280, height: 850}});
        const errors = [];
        page.on('pageerror', error => errors.push(String(error)));
        await page.goto('file://' + path.resolve(__dirname, '../ui.html'));
        await page.evaluate(() => {
            window.liveFixtureCalls = [];
            window.pywebview = {api: new Proxy({}, {get: (_, name) => async (...args) => {
                window.liveFixtureCalls.push([name, ...args]);
                if (name === 'get_kira_live_status') return {conversation_ready: true, checks: [], blockers: []};
                if (name === 'new_chat') return {id: 'ui-fixture'};
                if (name === 'start_live_voice') return {ok: true, chat_id: 'ui-fixture'};
                if (name === 'set_live_microphone_muted') return {ok: true, muted: args[0]};
                if (name === 'get_chat') return {id: 'ui-fixture', messages: []};
                if (name === 'stop_live_voice' || name === 'interrupt_live_response') return {ok: true};
                return [];
            }})};
            isBackendReady = true;
            currentChatId = 'ui-fixture';
            modelSelector.value = 'orchestrator';
            markKiraBootReady();
        });
        await page.evaluate(() => toggleLiveMonitor(true));
        assert.equal(await page.locator('#voice-live').getAttribute('aria-hidden'), 'false');
        assert.equal(await page.evaluate(() => liveVisual.renderer.running), true);
        await page.locator('#voice-listen-btn').click();
        assert.equal(await page.locator('#live-language').inputValue(), 'English');
        assert.equal(await page.evaluate(() => modelSelector.value), 'kira');
        await page.evaluate(() => handleVoiceBackendEvent({event:'FIRST_AUDIO',extra:{chat_id:'ui-fixture',turn_latency_seconds:3.25}}));
        assert.match(await page.locator('#live-latency').textContent(), /3.25s/);
        await page.evaluate(() => handleVoiceBackendEvent({event:'TURN_METRICS',extra:{chat_id:'ui-fixture',turn_to_audio_seconds:3.25,listener_seconds:.4,speech_start_seconds:.6,capture_overflows:0,playback_underflows:0}}));
        assert.match(await page.locator('#live-latency').textContent(), /Playback underruns 0/);
        await page.locator('#voice-listen-btn').click();
        assert.equal(await page.locator('#voice-listen-btn').getAttribute('aria-pressed'), 'true');
        await page.locator('#voice-listen-btn').click();
        assert.equal(await page.locator('#voice-listen-btn').getAttribute('aria-pressed'), 'false');
        await page.getByRole('button', {name: 'Interrupt KIRA', exact: true}).click();
        await page.getByRole('button', {name: 'Show transcript', exact: true}).click();
        assert.equal(await page.locator('#voice-live-transcript').evaluate(node => node.classList.contains('expanded')), true);
        await page.evaluate(() => appendVoiceTranscript('KIRA', 'Hello. What would you like to talk about?'));
        const transcriptOnce = await page.locator('#voice-live-transcript').textContent();
        await page.evaluate(() => appendVoiceTranscript('KIRA Live 1', 'Hello. What would you like to talk about?'));
        assert.equal(await page.locator('#voice-live-transcript').textContent(), transcriptOnce);
        require('node:fs').mkdirSync(path.resolve(__dirname, '../kira_voice_tmp'), {recursive: true});
        await page.screenshot({path: path.resolve(__dirname, '../kira_voice_tmp/live-ui-desktop.png')});
        await page.setViewportSize({width: 390, height: 844});
        await page.screenshot({path: path.resolve(__dirname, '../kira_voice_tmp/live-ui-mobile.png')});
        const overflow = await page.locator('.live-controls').evaluate(node => node.getBoundingClientRect().right > window.innerWidth);
        assert.equal(overflow, false);
        await page.getByRole('button', {name: 'End live conversation', exact: true}).click();
        assert.equal(await page.locator('#voice-live').getAttribute('aria-hidden'), 'true');
        assert.equal(await page.evaluate(() => modelSelector.value), 'orchestrator');
        assert.equal(await page.evaluate(() => liveVisual.renderer.running), false);
        const calls = await page.evaluate(() => window.liveFixtureCalls);
        assert(calls.some(call => call[0] === 'interrupt_live_response'));
        assert(calls.some(call => call[0] === 'start_live_voice' && call[2] === 'English'));
        assert(!calls.some(call => call[0] === 'start_voice_capture' || call[0] === 'stop_voice'));
        await page.evaluate(() => { renderKiraLiveReadiness({conversation_ready: false, checks: [], blockers: ['fixture blocker']}); });
        assert.equal(await page.locator('#voice-listen-btn').isDisabled(), true);
        assert.equal(errors.length, 0, errors.join('\n'));
        console.log('PASS: Live start/mute/resume/interrupt/transcript/end, timing/underrun display, readiness gate, desktop/mobile layout. Mock backend; not an audio test.');
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
