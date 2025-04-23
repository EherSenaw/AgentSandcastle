type ChatResponse = {
	response: string;
  };
  
  function appendMessage(content: string, sender: "user" | "bot"): void {
	const chatBox = document.getElementById("chat-box") as HTMLDivElement;
	const message = document.createElement("div");
	message.classList.add("message", sender);
	message.innerHTML = `<strong>${sender === "user" ? "You" : "LLM"}:</strong> ${content}`;
	chatBox.appendChild(message);
	chatBox.scrollTop = chatBox.scrollHeight;
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
  
  function setup(): void {
	const sendButton = document.getElementById("send-button") as HTMLButtonElement;
	sendButton.addEventListener("click", sendMessage);
  
	const input = document.getElementById("user-input") as HTMLInputElement;
	input.addEventListener("keypress", (event) => {
	  if (event.key === "Enter") sendMessage();
	});
  }
  
  window.addEventListener("DOMContentLoaded", setup);