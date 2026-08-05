# PR Draft: Enable DeepSeek reasoning (thinking) in pulsar

**Status:** Entwurf — lokal verifiziert auf Warpgate (DeepSeek-V4-Flash IQ2XXS, pulsar-serve auf Port 11435). Noch nicht als PR eröffnet.

## Problem (zweiteilig)

### 1. Server startet gar nicht mit aktuellen DeepSeek-V4-Flash GGUFs

`pulsar-serve` bricht beim Laden ab:

```
pulsar-serve: gguf metadata is missing  thinking
```

Ursache: `ChatMarkers::resolve` sucht die Reasoning-Marker hart als Literal `" thinking"` / `" response"`. Offizielle DeepSeek-V4-Flash-GGUFs (z.B. das `-imatrix-0731`-Quant) speichern die Marker aber **byte-encodiert** (`Ġthinking` = Space-Byte 0x20 + "thinking"), weil der Konverter die BPE-Byte-Fallback-Repräsentation übernimmt. Verifiziert per GGUF-Vocab-Dump:

```
' thinking': FEHLT
'Ġthinking': found at [2118]
'Ġresponse': found at [4256]
```

Das eingebettete `tokenizer.chat_template` des GGUFs bestätigt die offizielle Formel: `thinking_start_token = ' thinking'`, `thinking_end_token = ' response'`.

### 2. Reasoning war für DeepSeek hart abgeschaltet

Selbst mit dem alten, laufenden Server kam nie `reasoning_content` — der Tokenizer behandelte DeepSeek als Sonderfall mit **hartem** Thinking-Off:

- `ChatMarkers::detect` setzte `think: false` ("thinking stays off, ds4's default here")
- `open_assistant()` rendert für `ChatStyle::Deepseek` immer `assistant +  response` und **ignorierte `self.think` komplett** — der einzige Style, der das Flag nicht respektiert (GLM, KimiK3, MiniMax, ChatMl können alle denken)
- `opens_thinking()` / `reasoning_capable()` meldeten `false`, sodass der Server die Ausgabe nie in `reasoning_content` + `content` splitten würde

## Fix (crates/tokenizer/src/lib.rs)

1. **Marker robust auflösen** (detect, DeepSeek-Zweig): Literal zuerst, Fallback auf byte-encodierte Form:
   ```rust
   let thinking = t.find_token(" thinking")
       .or_else(|| t.find_token("Ġthinking"))
       .ok_or(Error::MissingKey(" thinking/Ġthinking"))?;
   let response = t.find_token(" response")
       .or_else(|| t.find_token("Ġresponse"))
       .ok_or(Error::MissingKey(" response/Ġresponse"))?;
   ```

2. **`think: true` als Default** (detect): DeepSeek-V4-Flash ship thinking ON by default; wer rohe Think-Token nicht rendern will, sendet pro Request `reasoning_effort: "none"` / `enable_thinking: false` (Server-Support existiert, main.rs:1414-1426).

3. **`open_assistant()` respektiert `think`**:
   ```rust
   ChatStyle::Deepseek => {
       let mut v = vec![self.assistant];
       v.push(if self.think { self.aux0 } else { self.aux1 });
       v
   }
   ```
   Thinking an → offener ` thinking`-Marker (Modell schließt mit ` response`); aus → wie bisher ` response`.

4. **`opens_thinking()` / `reasoning_capable()`**: DeepSeek aufnehmen, damit der Server via `split_open_think` (existiert für GLM, main.rs:1269) korrekt splittet.

5. **`render_assistant_history()` (DeepSeek-Zweig)**: History-Turns replayen in thinking-off-Form (bare ` response`), Reasoning wird nie zurückgespielt — dieselbe Policy wie Harmony ("the analysis channel is not replayed"). Das offizielle ds4-Template öffnet ` thinking` nur, wenn `reasoning_content` vorhanden ist; der Server sendet das für History nie. Ohne diesen Fix würden historische Turns mit offenem Think-Block gerendert.

## Verifikation (Warpgate, DeepSeek-V4-Flash IQ2XXS, Port 11435, GPU 3090)

| Test | Vorher | Nachher |
|---|---|---|
| Serverstart mit `-imatrix-0731`-GGUF | Crash `missing  thinking` | OK (Port offen nach ~22s) |
| Default-Request | nur `content`, kein Reasoning | `reasoning_content` + `content` getrennt |
| `enable_thinking: false` | — | nur `content` (Opt-out greift) |
| Multi-Turn (geheime Zahl 42) | — | Erinnerung funktioniert, Reasoning + korrekte Antwort |

Beispiel-Response (2+2, max_tokens 800):
```
reasoning_content: "We need answer simple. User asks \"2+2?\" Need compute 4. Final concise."
content: "4"
```

`cargo check -p tokenizer` und `cargo build --release -p serve` sauber.

## Offene Fragen / Review-Punkte

- **Default `think: true`** ändert das Default-Verhalten für alle DeepSeek-Nutzer. Chat-UIs ohne Reasoning-Rendering bekommen doppelte Felder; die meisten OpenAI-kompatiblen Clients (OpenWebUI, llama.cpp-Clients) handhaben `reasoning_content` korrekt, aber nicht alle. Alternative: Default `false`, nur per Request aktivierbar. Empfehlung: `true` — Reasoning ist das Feature des Modells.
- **Streaming:** Split nutzt `opens_thinking()` (main.rs:1580) — sollte mit Fix 4 automatisch funktionieren, müsste gegen den Stream-Endpoint verifiziert werden (lokal noch nicht getestet).
- **`Ġ`-Fallback betrifft nur DeepSeek-Zweig**: Andere Styles (ChatMl, GLM) suchen weiterhin nur Literale — falls deren GGUFs ebenfalls byte-encodiert exportiert werden, wäre das ein analoger Follow-up-Fix.
- **Byte-Encoding-Kommentar:** Die `Ġ`-Form ist die BPE-übliche Repräsentation (Space als `Ġ`); pulsars `gpt2_byte_to_char`-Map macht daraus im decoded Stream exakt `" thinking"`. Konsumenten sehen also weiterhin die offizielle Form.

## Diff-Stat

```
crates/tokenizer/src/lib.rs | ~40 Zeilen, 5 Stellen
```
