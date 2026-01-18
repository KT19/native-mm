import { useRef, useState } from "react";
import { Scene } from "./Scene";
import { Canvas } from "@react-three/fiber";
import { Send, ImagePlus, X } from "lucide-react";

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
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    // Reset file input so the same file can be uploaded again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
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
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"
              }`}
          >
            <div
              className={`max-w-[80%] p-4 rounded-2xl font-mono ${m.role === "user" ? "bg-blue-600 shadow-lg" : "bg-cyan-600"
                }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-700/80 backdrop-blur-sm text-slate-200 px-4 py-3 rounded-2xl animate-pulse text-sm">
              AI is thinking...
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <footer className="z-10 p-6 bg-linear-to-t from-slate-900/80 via-slate-900/40 to-transparent">
        <div className="max-w-4xl mx-auto relative group">
          {/* Image Preview */}
          {imagePreview && (
            <div className="absolute left-3 bottom-16 flex items-center gap-2 bg-slate-700/90 rounded-xl p-2 backdrop-blur-md">
              <img
                src={imagePreview}
                alt="Preview"
                className="h-12 w-12 object-cover rounded-lg"
              />
              <button
                onClick={() => {
                  setImage(null);
                  setImagePreview(null);
                }}
                className="p-1 bg-red-500/60 hover:bg-red-500 rounded-full transition"
              >
                <X size={14} />
              </button>
            </div>
          )}

          {/* Hidden File Input */}
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleImageUpload(file);
            }}
          />

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onPaste={handlePaste}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Type a message... (Shift+Enter for new line)"
            rows={1}
            className="w-full bg-slate-800/90 rounded-2xl px-5 py-4 pr-28 text-slate-100 placeholder:text-slate-400 outline-none resize-none backdrop-blur-md transition-all duration-200 focus:bg-slate-800"
            style={{ minHeight: "56px", maxHeight: "200px" }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "56px";
              target.style.height = Math.min(target.scrollHeight, 200) + "px";
            }}
          />

          {/* Buttons Container */}
          <div className="absolute right-3 bottom-3 flex items-center gap-2">
            {/* Image Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2.5 bg-slate-600/80 hover:bg-slate-500 rounded-xl transition-all duration-200 shadow-lg"
              title="Upload image"
            >
              <ImagePlus size={20} />
            </button>

            {/* Send Button */}
            <button
              onClick={sendMessage}
              disabled={loading || (!input.trim() && !image)}
              className="p-2.5 bg-linear-to-r from-blue-500 to-cyan-500 rounded-xl hover:from-blue-400 hover:to-cyan-400 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg hover:shadow-blue-500/25"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
