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
function appendMessage(content, sender) {
    const chatBox = document.getElementById("chat-box");
    const message = document.createElement("div");
    message.classList.add("message", sender);
    message.innerHTML = `<strong>${sender === "user" ? "You" : "LLM"}:</strong> ${content}`;
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
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
function setup() {
    const sendButton = document.getElementById("send-button");
    sendButton.addEventListener("click", sendMessage);
    const input = document.getElementById("user-input");
    input.addEventListener("keypress", (event) => {
        if (event.key === "Enter")
            sendMessage();
    });
}
window.addEventListener("DOMContentLoaded", setup);
