// Browser interaction checks use a deterministic native-API fixture, never models or personal chats.
// KIRA_PLAYWRIGHT_MODULE may point to an existing Playwright installation.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { chromium } = require(process.env.KIRA_PLAYWRIGHT_MODULE || 'playwright');
const output = process.env.KIRA_UI_SCREENSHOTS || '/tmp/kira-workspace-ui';
fs.mkdirSync(output, { recursive: true });

async function installFixture(page) {
    await page.evaluate(async () => {
        window.workspaceFixtureCalls = [];
        window.fixtureChats = [
            {id: 'research', title: 'Research for the next release', created_at: 1791139200},
            {id: "quote's-chat", title: '<script>alert(1)</script> planning', created_at: 1791139200}
        ];
        window.pywebview = {api: new Proxy({}, {get: (_, name) => async (...args) => {
            window.workspaceFixtureCalls.push([name, ...args]);
            if (name === 'get_chats') return window.fixtureChats;
            if (name === 'new_chat') return {id: 'new-workspace-chat'};
            if (name === 'load_chat') return {id: args[0], messages: [
                {role:'user', content:'Help me plan a focused release.'},
                {role:'assistant', content:'Start with the core workflow.\n\n**Plan**\n- Verify the composer.\n- Check keyboard navigation.\n- Review the release.', model:'kira'}
            ]};
            if (name === 'delete_chat') { window.fixtureChats = window.fixtureChats.filter(chat => chat.id !== args[0]); return {ok:true}; }
            if (name === 'send_prompt') return {chat_id: 'new-workspace-chat'};
            if (name === 'get_coding_settings') return {default_provider:'openai', providers:[{id:'openai',label:'OpenAI',configured:true,default_model:'fixture-model',models:['fixture-model']}]};
            if (name === 'get_coding_session') return {project_path:'/tmp/fixture-project', ide:'fixture', provider:'openai', model:'fixture-model', harness:'kira', ides:[{id:'fixture',label:'Fixture IDE',available:true}],harnesses:[{id:'kira',label:'KIRA Native',available:true}]};
            if (name === 'get_coding_budget') return {};
            if (name === 'save_coding_session') return {ok:true};
            if (name === 'get_kira_menu') return {sections:[]};
            if (name === 'get_integration_settings') return {mcp:{},plugins:[]};
            return [];
        }})};
        isBackendReady = true;
        currentChatId = 'new-workspace-chat';
        markKiraBootReady();
        await refreshChatHistory();
    });
    await page.locator('#boot-shell').waitFor({state: 'hidden'});
}

(async () => {
    const browser = await chromium.launch({headless: true, ...(process.env.KIRA_CHROME_PATH ? {executablePath: process.env.KIRA_CHROME_PATH} : {})});
    try {
        const page = await browser.newPage({viewport: {width:1440,height:960}, reducedMotion:'reduce'});
        const errors = [];
        page.on('pageerror', error => errors.push(String(error)));
        await page.goto('file://' + path.resolve(__dirname, '../ui.html'));
        await installFixture(page);
        assert.equal(await page.locator('#send-button').isDisabled(), true);
        await page.screenshot({path:path.join(output,'workspace-desktop.png')});
        await page.getByRole('button', {name:'Explore an idea'}).click();
        assert.equal(await page.locator('#user-input').inputValue(), 'Help me research ');
        assert.equal(await page.locator('#send-button').isDisabled(), false);
        assert(!await page.evaluate(() => workspaceFixtureCalls.some(call => call[0] === 'send_prompt')));
        await page.locator('#user-input').fill('A considered next step');
        await page.locator('#user-input').press('Shift+Enter');
        assert.match(await page.locator('#user-input').inputValue(), /\n$/);
        await page.getByRole('button', {name:'Send message',exact:true}).click();
        assert(await page.evaluate(() => workspaceFixtureCalls.some(call => call[0] === 'send_prompt' && call[1] === 'A considered next step')));
        assert.equal(await page.locator('#send-button').isDisabled(), true);
        assert.equal(await page.locator('#welcome-block').count(), 0);
        await page.evaluate(() => {removeLoadingIndicator();renderAIMessage('A clear next step starts here.','KIRA Live 1');});
        assert.equal(await page.locator('.message-user').count(),1);
        assert.equal(await page.locator('.message-assistant').count(),1);
        await page.screenshot({path:path.join(output,'workspace-conversation.png')});

        await page.keyboard.press('Meta+k');
        assert.equal(await page.locator('#workspace-search-modal').isVisible(), true);
        assert.equal(await page.evaluate(() => document.activeElement.id),'workspace-search');
        await page.locator('#workspace-search').fill('research');
        assert.equal(await page.locator('.search-result').count(),1);
        await page.locator('#workspace-search').press('ArrowDown');
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#workspace-search-modal').isVisible(),false);
        assert(await page.evaluate(() => workspaceFixtureCalls.some(call => call[0] === 'load_chat' && call[1] === 'research')));
        await page.keyboard.press('Meta+k');
        await page.locator('#workspace-search').fill('no-such-result');
        assert.equal(await page.getByText('No matches. Try another chat title or action.').isVisible(),true);
        await page.locator('#workspace-search').fill('');
        await page.locator('#workspace-search').press('Shift+Tab');
        assert.equal(await page.evaluate(() => document.activeElement.className),'search-result');
        await page.keyboard.press('Tab');
        assert.equal(await page.evaluate(() => document.activeElement.id),'workspace-search');
        await page.screenshot({path:path.join(output,'workspace-search.png')});
        await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(() => document.querySelector('main').inert), false);
        assert.equal(await page.locator('#chat-history script').count(),0);
        await page.getByRole('button',{name:'Delete <script>alert(1)</script> planning',exact:true}).click();
        assert(await page.evaluate(() => workspaceFixtureCalls.some(call => call[0] === 'delete_chat' && call[1] === "quote's-chat")));

        await page.keyboard.press('Meta+b');
        assert.equal(await page.locator('#workspace-sidebar').isVisible(),false);
        await page.reload();
        await installFixture(page);
        assert.equal(await page.locator('#workspace-sidebar').isVisible(),false);
        await page.keyboard.press('Meta+b');
        assert.equal(await page.locator('#workspace-sidebar').isVisible(),true);
        await page.keyboard.press('Meta+n');
        await page.locator('#welcome-block').waitFor();
        await page.getByRole('button',{name:'Build with code'}).click();
        assert.equal(await page.locator('#coding-toolbar').isVisible(),true);
        assert.equal(await page.locator('#welcome-greeting').textContent(),'What shall we build?');
        assert.equal(await page.locator('#composer-context').textContent(),'Code workspace');
        await page.screenshot({path:path.join(output,'workspace-code.png')});
        await page.locator('[data-mode="agent"]').click();
        assert.equal(await page.locator('#coding-toolbar').isVisible(),false);
        await page.locator('#user-input').fill('');
        await page.evaluate(() => {pendingAttachments=[{name:'brief.md',path:'/tmp/brief.md',size:48}];renderAttachmentTray();});
        assert.equal(await page.locator('#send-button').isDisabled(),false);
        await page.getByRole('button',{name:'Remove brief.md',exact:true}).click();
        assert.equal(await page.locator('#send-button').isDisabled(),true);
        await page.getByRole('button',{name:'Browse slash commands'}).click();
        assert.equal(await page.locator('#user-input').inputValue(),'/');
        assert.equal(await page.locator('#slash-panel').isVisible(),true);
        await page.locator('#user-input').fill('');

        for (const width of [320,390,768,900]) {
            await page.setViewportSize({width,height:844});
            await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            const bounds = await page.evaluate(() => ({width:innerWidth,body:document.body.scrollWidth,composer:document.getElementById('composer-wrap').getBoundingClientRect().toJSON(),header:document.querySelector('header').getBoundingClientRect().toJSON()}));
            assert(bounds.body <= width, JSON.stringify(bounds));
            assert(bounds.composer.right <= width && bounds.composer.left >= 0, JSON.stringify(bounds));
            assert(bounds.composer.bottom <= 844, JSON.stringify(bounds));
            assert.equal(bounds.header.top,0);
        }
        await page.setViewportSize({width:390,height:844});
        await page.locator('#sidebar-toggle').click();
        assert.equal(await page.locator('#workspace-sidebar').isVisible(),true);
        assert.equal(await page.evaluate(() => document.querySelector('main').inert),true);
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('#workspace-sidebar').isVisible(),false);
        assert.equal(await page.evaluate(() => document.querySelector('main').inert),false);
        await page.locator('#sidebar-toggle').click();
        await page.getByRole('button',{name:'Search chats and actions',exact:true}).first().click();
        assert.equal(await page.locator('#workspace-search-modal').isVisible(),true);
        await page.keyboard.press('Escape');
        await page.evaluate(() => clearChat());
        await page.screenshot({path:path.join(output,'workspace-mobile.png')});
        await page.setViewportSize({width:390,height:600});
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const shortBounds=await page.locator('#composer-wrap').boundingBox();
        assert(shortBounds.y+shortBounds.height<=600,JSON.stringify(shortBounds));
        assert.equal(errors.length,0,errors.join('\n'));
        console.log('PASS: composer/send, suggestions, chat search/keyboard/focus trap, safe chat titles/delete, persisted sidebar, code mode, attachments, slash commands, mobile drawer and responsive layouts. Fixture backend; native model/audio behavior is not exercised.');
        console.log('Screenshots: '+output);
    } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
