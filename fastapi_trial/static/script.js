"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
// Synchronized version
function appendMessage(content, sender) {
    const chatBox = document.getElementById("chat-box");
    const message = document.createElement("div");
    message.classList.add("message", sender);
    message.innerHTML = `<strong>${sender === "user" ? "You" : "LLM"}:</strong> ${content}`;
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
}
// Async version
function streamMessage(userText) {
    return __awaiter(this, void 0, void 0, function* () {
        const chatBox = document.getElementById("chat-box");
        const messageDiv = document.createElement("div");
        const answerDiv = document.createElement("div");
        messageDiv.classList.add("message", "bot");
        answerDiv.classList.add("message", "bot", "final");
        messageDiv.innerHTML = `<strong>LLM(thinking):</strong> <span class="stream"></span>`;
        answerDiv.innerHTML = `<strong>LLM:</strong> <span class="final-pre"></span>`;
        chatBox.appendChild(messageDiv);
        chatBox.appendChild(answerDiv);
        const streamTarget = messageDiv.querySelector(".stream");
        const finalTarget = answerDiv.querySelector(".final-pre");
        // 1) Kick off the POST.
        const response = yield fetch("/chat-stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_input: userText })
        });
        if (!response.body) {
            streamTarget.textContent = "⚠️ Streaming not supported by this browser";
            finalTarget.textContent = "️️Failed to stream.";
            return;
        }
        // 2) Read chunks from the response stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { done, value } = yield reader.read();
            if (done)
                break;
            buffer += decoder.decode(value, { stream: true });
            // split on newlines; keep any partial line in buffer
            const parts = buffer.split("\n");
            buffer = parts.pop();
            let prev_stream = streamTarget.textContent;
            for (const line of parts) {
                if (!line.trim())
                    continue;
                // {"type":..., "data":...}
                const msg = JSON.parse(line);
                if (msg.type.startsWith("thought")) {
                    if (msg.type.endsWith("intermediate")) {
                        // append each though-token to the same "stream" span.
                        streamTarget.textContent += msg.data;
                    }
                    else if (msg.type.endsWith("answer")) {
                        streamTarget.textContent = (prev_stream + "\n\n" + msg.data);
                        prev_stream = streamTarget.textContent;
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                }
                else if (msg.type.startsWith("final")) {
                    if (msg.type.endsWith("intermediate")) {
                        finalTarget.textContent += msg.data;
                    }
                    else if (msg.type.endsWith("answer")) {
                        finalTarget.textContent = msg.data;
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                }
                // chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    });
}
function sendMessage() {
    return __awaiter(this, void 0, void 0, function* () {
        const input = document.getElementById("user-input");
        const userText = input.value.trim();
        if (!userText)
            return;
        appendMessage(userText, "user");
        input.value = "";
        try {
            const response = yield fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ user_input: userText }),
            });
            const data = yield response.json();
            appendMessage(data.response, "bot");
        }
        catch (error) {
            appendMessage("⚠️ Error: Could not reach backend.", "bot");
            console.error(error);
        }
    });
}
function sendMessage_async() {
    return __awaiter(this, void 0, void 0, function* () {
        const input = document.getElementById("user-input");
        const userText = input.value.trim();
        if (!userText)
            return;
        appendMessage(userText, "user");
        input.value = "";
        try {
            streamMessage(userText);
        }
        catch (error) {
            streamMessage("⚠️ Error: Could not reach backend.");
            console.error(error);
        }
    });
}
function setup() {
    const sendButton = document.getElementById("send-button");
    sendButton.addEventListener("click", sendMessage);
    const input = document.getElementById("user-input");
    input.addEventListener("keypress", (event) => {
        if (event.key === "Enter")
            sendMessage();
    });
}
function setup_async() {
    const sendButton = document.getElementById("send-button");
    sendButton.addEventListener("click", sendMessage_async);
    const input = document.getElementById("user-input");
    input.addEventListener("keypress", (event) => {
        if (event.key === "Enter")
            sendMessage_async();
    });
}
window.addEventListener("DOMContentLoaded", setup_async);
