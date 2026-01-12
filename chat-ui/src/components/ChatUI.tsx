import { useRef, useState } from "react";
import { Scene } from "./Scene";
import { Canvas } from "@react-three/fiber";
import { Send } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  //Upload
  const handleImageUpload = (file: File) => {
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
  };
  //Handle clipboard
  const handlePaste = (e: React.ClipboardEvent) => {
    const item = e.clipboardData.items[0];
    if (item?.type.includes("image")) {
      const file = item.getAsFile();
      if (file) handleImageUpload(file);
    }
  };

  const clearHistory = () => {
    setMessages([]);
    setImage(null);
    setImagePreview(null);
  };

  const sendMessage = async () => {
    if (!input.trim() && !image) return;

    const currentInput = input;
    const newMessages: Message[] = [
      ...messages,
      { role: "user", content: input },
    ];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    //Prepare Multipart Data
    const formData = new FormData();
    if (image) {
      formData.append("image", image);
    }
    formData.append("user_text", currentInput);
    formData.append("history", JSON.stringify(messages));

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (err) {
      console.error("Inference Error:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Error: Cound not connect to the backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative h-screen w-full bg-slate-400 text-slate-100 overflow-hidden flex flex-col">
      {/* R3F background */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <Canvas>
          <Scene imageUrl={imagePreview} />
        </Canvas>
      </div>

      {/* UI Overlay */}
      <header className="z-10 p-4 border-b border-zinc-400 bg-slate-900/50 backdrop-blur-md flex justify-between items-center">
        <h1 className="text-xl font-bold tracking-tight">NMM Chat</h1>
        <button
          onClick={clearHistory}
          className="px-3 py-1 text-sm bg-red-500/40 hover:bg-red-500 rounded transition"
        >
          Clear Session
        </button>
      </header>

      {/* Chat History */}
      <div
        className="flex-1 overflow-y-auto z-10 p-6 space-y-4"
        ref={scrollRef}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${
              m.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] p-4 rounded-2xl font-mono ${
                m.role === "user" ? "bg-blue-600 shadow-lg" : "bg-cyan-600"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-slate-400 animate-pulse text-sm">
            AI is thinking...
          </div>
        )}
      </div>

      {/* Input Area */}
      <footer className="z-10 p-6 bg-linear-to-t from-slate-400 to-transparent">
        <div className="max-w-4xl mx-auto relative group">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onPaste={handlePaste}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
            placeholder="Type a message..."
            className="w-full bg-slate-600/80 border border-white/10 rounded-2xl p-4 pr-12 focus:ring-2 focus:ring-blue-500 outline-none resize-none h24 backdrop-blur-sm"
          />
          <button
            onClick={sendMessage}
            className="absolute right-4 bottom-6 p-2 bg-blue-500 rounded-xl hover:bg-blue-400 transition"
          >
            <Send />
          </button>
        </div>
      </footer>
    </div>
  );
}
