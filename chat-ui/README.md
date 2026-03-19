# NMM Chat Interface

A modern, responsive web interface for interacting with the Native Multimodal Model. Built with React, Vite, and TailwindCSS, this UI provides an intuitive experience for multimodal chat.

## Getting Started

### Installation

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`.

## 🔗 Connecting to the Backend

The UI expects a running backend server from the `native-mm` package. By default, it communicates with the local server scripts provided in the core repository.

To start the backend server:

```bash
cd ../native-mm
uv run scripts/server_chat.py
```
