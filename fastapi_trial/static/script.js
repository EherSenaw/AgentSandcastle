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
// Synchronized version - Modified to handle empty content for user (e.g. image only)
function appendMessage(content, sender) {
    const chatBox = document.getElementById("chat-box");
    const message = document.createElement("div");
    message.classList.add("message", sender);
    if (sender === "user" && content === "") {
        // Display a placeholder if user sends an image without text
        message.innerHTML = `<strong>You:</strong> [Image Sent]`;
    }
    else {
        message.innerHTML = `<strong>${sender === "user" ? "You" : "LLM"}:</strong> ${content}`;
    }
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
}
// Async version - Modified to accept FormData
function streamMessage(formData) {
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
            // headers: { "Content-Type": "application/json" }, // REMOVED for FormData
            body: formData // Use FormData directly
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
            // This is the non-streaming version, also needs FormData if used with multimodal.
            // For now, focusing on streaming path as per instructions.
            // If this path is also needed, it should be updated similarly to streamMessage.
            const response = yield fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json", // Keep for non-streaming if it only accepts JSON
                },
                body: JSON.stringify({ user_input: userText }), // This would need to change if /chat supports FormData
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
        const imageInput = document.getElementById("image-input");
        const imageFile = imageInput.files ? imageInput.files[0] : null;
        if (!userText && !imageFile)
            return; // Only proceed if there's text or an image
        // Display user's message (text or placeholder for image)
        // The appendMessage function was modified to handle empty userText if an image is present.
        appendMessage(userText, "user");
        input.value = ""; // Clear text input
        if (imageInput)
            imageInput.value = ""; // Clear file input
        const formData = new FormData();
        formData.append("user_input", userText); // Send empty string if no text, backend handles it
        if (imageFile) {
            formData.append("image_file", imageFile);
            // Optional: Display image thumbnail in chat
            // This is a simplified way to show the image in the chat right after selection.
            // It finds the last message (assumed to be the one just appended) and adds an img tag.
            const reader = new FileReader();
            reader.onload = (e) => {
                var _a;
                const imgElement = document.createElement("img");
                imgElement.src = (_a = e.target) === null || _a === void 0 ? void 0 : _a.result;
                imgElement.style.maxWidth = "200px";
                imgElement.style.maxHeight = "200px";
                imgElement.style.display = "block";
                imgElement.style.marginTop = "5px"; // Some spacing
                const chatBox = document.getElementById("chat-box");
                const userMessages = chatBox.querySelectorAll(".message.user");
                if (userMessages.length > 0) {
                    const lastUserMessage = userMessages[userMessages.length - 1];
                    lastUserMessage.appendChild(imgElement); // Append image to the user's message div
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            };
            reader.readAsDataURL(imageFile);
        }
        try {
            // Pass formData to streamMessage
            streamMessage(formData);
        }
        catch (error) {
            // This catch block might be for errors setting up the stream, not for fetch errors within streamMessage
            appendMessage("⚠️ Error: Could not initiate stream.", "bot"); // Use appendMessage for consistency
            console.error(error);
        }
    });
}
function setup() {
    const sendButton = document.getElementById("send-button");
    // sendButton.addEventListener("click", sendMessage); // Original non-streaming sendMessage
    // Ensure event listeners are correctly targeting the intended functions.
    // If only async streaming is used, this might not be needed or should call sendMessage_async.
    // For now, keeping it as it was but commented out the click listener if focusing on async.
    // If sendMessage is also to be multimodal, it needs updating and this listener re-enabled.
    const input = document.getElementById("user-input");
    // input.addEventListener("keypress", (event) => {
    // 	if (event.key === "Enter") sendMessage(); 
    // });
}
function setup_async() {
    const sendButton = document.getElementById("send-button");
    sendButton.addEventListener("click", sendMessage_async);
    const input = document.getElementById("user-input");
    // Also listen to Enter key on the image input if desired, though typically Enter on text input is primary.
    // const imageInput = document.getElementById("image-input") as HTMLInputElement; 
    input.addEventListener("keypress", (event) => {
        if (event.key === "Enter")
            sendMessage_async();
    });
    // Example: if you want Enter on image input to also submit (less common)
    // imageInput.addEventListener("keypress", (event) => {
    //     if (event.key === "Enter") sendMessage_async();
    // });
}
window.addEventListener("DOMContentLoaded", setup_async); // Changed to setup_async to use streaming by default
