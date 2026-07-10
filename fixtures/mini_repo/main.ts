import { createServer } from "http";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

const PORT = parseInt(process.env.PORT || "3000", 10);
const CONFIG_PATH = process.env.CONFIG_PATH || "./config.yaml";

interface AppConfig {
  host: string;
  port: number;
  debug: boolean;
  database: {
    url: string;
    poolSize: number;
  };
}

function loadConfig(): AppConfig {
  if (!existsSync(CONFIG_PATH)) {
    console.warn(`Config not found at ${CONFIG_PATH}, using defaults`);
    return {
      host: "0.0.0.0",
      port: PORT,
      debug: false,
      database: { url: "sqlite:///local_dev.db", poolSize: 10 },
    };
  }
  const raw = readFileSync(CONFIG_PATH, "utf-8");
  return parseYaml(raw);
}

function parseYaml(raw: string): AppConfig {
  const lines = raw.split("\n");
  const config: Record<string, any> = {};
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#") || trimmed === "") continue;
    const [key, ...rest] = trimmed.split(":").map((s) => s.trim());
    config[key] = rest.join(":").replace(/['"]/g, "");
  }
  return config as unknown as AppConfig;
}

const server = createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "ok", service: "ai-knowledge-bot" }));
});

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
