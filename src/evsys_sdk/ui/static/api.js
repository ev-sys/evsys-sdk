// Tiny fetch wrapper for the local JSON API (same-origin, read-only).
async function apiGet(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).error || detail; } catch (e) {}
    throw new Error(`${path} → ${res.status} ${detail}`);
  }
  return res.json();
}
