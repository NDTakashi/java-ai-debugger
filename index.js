import express from "express";
import http from "http";
import { Server } from "socket.io";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest";
dotenv.config();

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: [
      "http://localhost:5173",
      "https://java-ai-debugger-frontendcopy.vercel.app",
    ],
    methods: ["GET", "POST"],
  },
});

app.use(cors());

const COLLECTION = "java_book";

// ─── Clients ──────────────────────────────────────────────────────────────────
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});

// const qdrant = new QdrantClient({ url: "http://localhost:6333" });

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// ─── Verify Qdrant is reachable and collection exists ────────────────────────
async function verifyCollection() {
  const { collections } = await qdrant.getCollections();
  const exists = collections.some((c) => c.name === COLLECTION);
  if (!exists) {
    console.error(
      `Qdrant collection "${COLLECTION}" not found. Run: node ingest.js`,
    );
    process.exit(1);
  }
  const info = await qdrant.getCollection(COLLECTION);
  console.log(
    `Qdrant ready — "${COLLECTION}" has ${info.points_count} vectors`,
  );
}

// ─── Embed a single query string ─────────────────────────────────────────────
async function embedQuery(text) {
  const res = await openai.embeddings.create({
    model: "openai/text-embedding-3-small",
    input: text,
  });
  return res.data[0].embedding;
}

// ─── Retrieve top-k semantically similar chunks ───────────────────────────────
async function retrieve(query, k = 4) {
  const vector = await embedQuery(query);
  const results = await qdrant.search(COLLECTION, {
    vector,
    limit: k,
    with_payload: true,
    with_vector: false,
  });
  // return full metadata alongside text and score
  return results.map((r) => ({
    text: r.payload.text,
    score: r.score,
    type: r.payload.type || "text",
    page: r.payload.page || null,
    chapter: r.payload.chapter || null,
  }));
}

// ─── Conversational detection (same as before) ────────────────────────────────
const CASUAL_PATTERNS = [
  /^(hi|hey|hello|howdy|sup|yo)\b/i,
  /^(thanks|thank you|thx|ty)\b/i,
  /^(bye|goodbye|see you|cya)\b/i,
  /^(ok|okay|got it|understood|makes sense|cool|great|nice|awesome)\b/i,
  /^(yes|no|nope|yep|sure|alright)\b/i,
  /^(who are you|what are you|what can you do|help)\b/i,
];
const NON_JAVA_TOPICS = [
  "time",
  "date",
  "weather",
  "news",
  "score",
  "stock",
  "price",
  "recipe",
  "movie",
  "sport",
  "capital",
  "country",
  "president",
  "population",
  "birthday",
  "football",
  "cricket",
  "music",
  "song",
  "actor",
  "politics",
];
const JAVA_ROOTS = [
  "java",
  "class",
  "object",
  "method",
  "interfac",
  "exception",
  "thread",
  "array",
  "loop",
  "string",
  "variabl",
  "compil",
  "runtim",
  "packag",
  "inherit",
  "polymorphi",
  "abstract",
  "generic",
  "lambda",
  "stream",
  "constructor",
  "overrid",
  "overload",
  "encapsul",
  "static",
  "void",
  "import",
  "extend",
  "implement",
  "switch",
  "instanceof",
];
const STOP_WORDS = new Set([
  "a",
  "an",
  "the",
  "is",
  "it",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "and",
  "or",
  "but",
  "not",
  "with",
  "this",
  "that",
  "these",
  "was",
  "are",
  "be",
  "been",
  "have",
  "has",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "can",
  "what",
  "how",
  "why",
  "when",
  "where",
  "who",
  "which",
  "i",
  "you",
  "we",
  "they",
  "me",
  "him",
  "her",
  "us",
  "them",
  "so",
  "if",
  "as",
  "by",
  "from",
  "just",
  "also",
  "more",
  "some",
  "any",
  "all",
  "no",
  "there",
  "get",
  "use",
  "used",
]);

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w));
}

function isConversational(message) {
  const trimmed = message.trim();
  const tokens = tokenize(trimmed);
  if (CASUAL_PATTERNS.some((p) => p.test(trimmed))) return true;
  if (trimmed.length < 20 && tokens.length === 0) return true;
  const lower = trimmed.toLowerCase();
  const hasNonJava = NON_JAVA_TOPICS.some((t) => lower.includes(t));
  const hasJavaToken = tokens.some((t) =>
    JAVA_ROOTS.some((r) => t.startsWith(r)),
  );
  if (hasNonJava && !hasJavaToken) return true;
  return false;
}

// ─── Socket.IO ────────────────────────────────────────────────────────────────
io.on("connection", (socket) => {
  console.log("User connected");

  socket.on("user-message", async (message) => {
    try {
      // Route 1 — casual chat, no retrieval needed
      if (isConversational(message)) {
        const completion = await openai.chat.completions.create({
          model: "poolside/laguna-m.1:free",
          messages: [
            {
              role: "system",
              content: `You are a friendly Java programming assistant.
- Greetings / small talk: reply warmly and briefly.
- General knowledge questions (time, weather etc): answer normally from your own knowledge.
- What can you do: explain you answer Java questions from an uploaded Java reference book.`,
            },
            { role: "user", content: message },
          ],
        });
        socket.emit("ai-response", completion.choices[0].message.content);
        return;
      }

      // Route 2 — Java query: embed → semantic search → LLM
      console.log(`Query: "${message}"`);
      const retrieved = await retrieve(message, 4);

      // emit chunks immediately so UI can show them while LLM is thinking
      socket.emit(
        "debug-chunks",
        retrieved.map((r, i) => ({
          index: i + 1,
          text: r.text,
          score: r.score,
          type: r.type,
          page: r.page,
          chapter: r.chapter,
        })),
      );

      // build context — wrap code chunks in fences so LLM treats them as code
      const context = retrieved
        .map((r, i) => {
          const header = `[Excerpt ${i + 1}]${r.chapter ? ` — ${r.chapter}` : ""}${r.page ? ` (page ${r.page})` : ""}:`;
          const body =
            r.type === "code" ? "```java\n" + r.text + "\n```" : r.text;
          return header + "\n" + body;
        })
        .join("\n\n");

      const completion = await openai.chat.completions.create({
        model: "poolside/laguna-m.1:free",
        messages: [
          {
            role: "system",
            content: `You are a Java programming assistant and code debugger.
Answer using the provided excerpts from the Java reference book.
- Answer ONLY from the excerpts.
- If not covered, say: "I could not find this in the uploaded document."
- For code questions, explain clearly using examples from the context.`,
          },
          {
            role: "user",
            content: `Excerpts from the Java book:\n\n${context}\n\nQuestion: ${message}`,
          },
        ],
      });

      socket.emit("ai-response", completion.choices[0].message.content);
    } catch (err) {
      console.error(err);
      socket.emit("ai-response", "An error occurred. Please try again.");
    }
  });

  socket.on("disconnect", () => console.log("User disconnected"));
});

// ─── Start ────────────────────────────────────────────────────────────────────
verifyCollection().then(() => {
  server.listen(5000, () =>
    console.log("Server running on http://localhost:5000"),
  );
});
