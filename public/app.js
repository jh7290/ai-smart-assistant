const configEl = document.querySelector("#config");
const conversationsEl = document.querySelector("#conversations");
const messagesEl = document.querySelector("#messages");
const titleInput = document.querySelector("#titleInput");
const personaInput = document.querySelector("#personaInput");
const memoryInput = document.querySelector("#memoryInput");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const newChatBtn = document.querySelector("#newChatBtn");
const saveSettingsBtn = document.querySelector("#saveSettingsBtn");

let currentConversationId = null;
let conversations = [];

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok && !data.fallback) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;"
  })[char]);
}

async function loadConfig() {
  const config = await api("/api/config");
  configEl.innerHTML = `
    <strong>${escapeHtml(config.provider)}</strong><br>
    ${escapeHtml(config.model || "未设置模型")}<br>
    ${config.configured ? "已配置 API Key" : "未配置 API Key，当前可本地演示"}
  `;
}

function renderConversations() {
  conversationsEl.innerHTML = conversations.map((item) => `
    <button class="conversation ${item.id === currentConversationId ? "active" : ""}" data-id="${item.id}">
      <strong>${escapeHtml(item.title)}</strong>
      <span>${new Date(item.updated_at).toLocaleString()}</span>
    </button>
  `).join("");
}

async function loadConversations() {
  const data = await api("/api/conversations");
  conversations = data.conversations;
  currentConversationId = currentConversationId || conversations[0]?.id;
  renderConversations();
  if (currentConversationId) {
    await loadMessages(currentConversationId);
  }
}

async function loadMessages(id) {
  const data = await api(`/api/conversations/${id}/messages`);
  currentConversationId = id;
  titleInput.value = data.conversation.title;
  personaInput.value = data.conversation.persona;
  memoryInput.value = data.conversation.memory;
  messagesEl.innerHTML = data.messages.length
    ? data.messages.map(renderMessage).join("")
    : `<div class="message assistant">你好，我已经准备好了。你可以让我帮你优化项目、写代码、拆需求或准备面试回答。</div>`;
  messagesEl.scrollTop = messagesEl.scrollHeight;
  renderConversations();
}

function renderMessage(item) {
  return `
    <article class="message ${item.role}">
      <div class="meta">${item.role === "user" ? "你" : "AI 助手"}</div>
      ${escapeHtml(item.content)}
    </article>
  `;
}

conversationsEl.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-id]");
  if (!button) return;
  await loadMessages(button.dataset.id);
});

newChatBtn.addEventListener("click", async () => {
  const data = await api("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "新对话" })
  });
  currentConversationId = data.id;
  await loadConversations();
});

saveSettingsBtn.addEventListener("click", async () => {
  if (!currentConversationId) return;
  await api(`/api/conversations/${currentConversationId}/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: titleInput.value,
      persona: personaInput.value,
      memory: memoryInput.value
    })
  });
  await loadConversations();
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || !currentConversationId) return;

  messagesEl.insertAdjacentHTML("beforeend", renderMessage({ role: "user", content: message }));
  messagesEl.insertAdjacentHTML("beforeend", `<article class="message assistant" id="pending">正在思考...</article>`);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  messageInput.value = "";

  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversationId: currentConversationId, message })
    });
    document.querySelector("#pending")?.remove();
    messagesEl.insertAdjacentHTML("beforeend", renderMessage({
      role: "assistant",
      content: `${data.fallback || data.answer}\n\n模式：${data.mode === "llm" ? "真实模型" : "本地兜底"}`
    }));
    await loadConversations();
  } catch (error) {
    document.querySelector("#pending")?.remove();
    messagesEl.insertAdjacentHTML("beforeend", renderMessage({ role: "assistant", content: error.message }));
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
});

Promise.all([loadConfig(), loadConversations()]).catch((error) => {
  configEl.textContent = error.message;
});
