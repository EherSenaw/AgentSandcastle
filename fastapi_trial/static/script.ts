type ChatResponse = {
	response: string;
};

// Synchronized version
function appendMessage(content: string, sender: "user" | "bot"): void {
	const chatBox = document.getElementById("chat-box") as HTMLDivElement;
	const message = document.createElement("div");
	message.classList.add("message", sender);
	message.innerHTML = `<strong>${sender === "user" ? "You" : "LLM"}:</strong> ${content}`;
	chatBox.appendChild(message);
	chatBox.scrollTop = chatBox.scrollHeight;
}
// Async version
async function streamMessage(userText: string): Promise<void> {
	const chatBox = document.getElementById("chat-box") as HTMLDivElement;
	const messageDiv = document.createElement("div");
	const answerDiv = document.createElement("div");
	messageDiv.classList.add("message", "bot");
	answerDiv.classList.add("message", "bot", "final");
	messageDiv.innerHTML = `<strong>LLM(thinking):</strong> <span class="stream"></span>`;
	answerDiv.innerHTML = `<strong>LLM:</strong> <span class="final-pre"></span>`;
	chatBox.appendChild(messageDiv);
	chatBox.appendChild(answerDiv);

	const streamTarget = messageDiv.querySelector(".stream") as HTMLSpanElement;
	const finalTarget = answerDiv.querySelector(".final-pre") as HTMLSpanElement;

	// 1) Kick off the POST.
	const response = await fetch("/chat-stream", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ user_input: userText })
	});
	if (!response.body) {
		streamTarget.textContent = "⚠️ Streaming not supported by this browser";
		finalTarget.textContent = "️️Failed to stream."
		return;
	}

	// 2) Read chunks from the response stream
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });

		// split on newlines; keep any partial line in buffer
		const parts = buffer.split("\n");
		buffer = parts.pop()!;

		let prev_stream = streamTarget.textContent;

		for (const line of parts) {
			if (!line.trim()) continue;
			// {"type":..., "data":...}
			const msg = JSON.parse(line) as { type: string; data: string };
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
}

async function sendMessage(): Promise<void> {
	const input = document.getElementById("user-input") as HTMLInputElement;
	const userText = input.value.trim();
	if (!userText) return;

	appendMessage(userText, "user");
	input.value = "";

	try {
		const response = await fetch("/chat", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ user_input: userText }),
		});

		const data: ChatResponse = await response.json();
		appendMessage(data.response, "bot");
	} catch (error) {
		appendMessage("⚠️ Error: Could not reach backend.", "bot");
		console.error(error);
	}
}

async function sendMessage_async(): Promise<void> {
	const input = document.getElementById("user-input") as HTMLInputElement;
	const userText = input.value.trim();
	if (!userText) return;

	appendMessage(userText, "user");
	input.value = "";

	try {
		streamMessage(userText);
	} catch (error) {
		streamMessage("⚠️ Error: Could not reach backend.");
		console.error(error);
	}
}

function setup(): void {
	const sendButton = document.getElementById("send-button") as HTMLButtonElement;
	sendButton.addEventListener("click", sendMessage);

	const input = document.getElementById("user-input") as HTMLInputElement;
	input.addEventListener("keypress", (event) => {
		if (event.key === "Enter") sendMessage();
	});
}

function setup_async(): void {
	const sendButton = document.getElementById("send-button") as HTMLButtonElement;
	sendButton.addEventListener("click", sendMessage_async);

	const input = document.getElementById("user-input") as HTMLInputElement;
	input.addEventListener("keypress", (event) => {
		if (event.key === "Enter") sendMessage_async();
	});
}

window.addEventListener("DOMContentLoaded", setup_async);