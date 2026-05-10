import React, { useMemo, useState } from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import {
  Activity,
  AlertTriangle,
  Camera,
  ClipboardList,
  Download,
  FileText,
  FlaskConical,
  HeartPulse,
  MessageCircle,
  ShieldCheck,
  Upload,
} from "https://esm.sh/lucide-react@0.468.0?deps=react@18.3.1";

const recommendedTests = [
  "CBC with hemoglobin",
  "Ferritin and iron profile",
  "Vitamin D",
  "Vitamin B12",
  "TSH / thyroid profile",
  "Zinc",
  "Fasting glucose or HbA1c if clinically relevant",
  "Hormonal tests only if advised by a doctor, especially for irregular periods, acne, or excess facial hair",
];

const initialProfile = {
  subject: "Self",
  name: "",
  age: 28,
  gender: "Female",
  hair_fall_level: 5,
  duration: "Less than 1 month",
  pattern: "Overall shedding",
  dandruff: "None",
  scalp_symptoms: "None",
  stress: 5,
  sleep_hours: 7,
  diet_quality: "Good",
  family_history: "No",
  chemical_treatments: "No",
  recent_illness: "No",
  medications: "",
  notes: "",
};

function App() {
  const [profile, setProfile] = useState(initialProfile);
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [labResult, setLabResult] = useState(null);
  const [report, setReport] = useState(null);
  const [chat, setChat] = useState([]);
  const [message, setMessage] = useState("");
  const [errors, setErrors] = useState([]);
  const [busy, setBusy] = useState("");

  const firstName = useMemo(() => profile.name.trim() || "there", [profile.name]);

  const update = (key, value) => setProfile((current) => ({ ...current, [key]: value }));

  const onImage = (file) => {
    setImage(file);
    setErrors([]);
    setPreview(file ? URL.createObjectURL(file) : "");
  };

  const analyze = async () => {
    const validation = [];
    if (!profile.name.trim()) validation.push("Name is required.");
    if (!image) validation.push("Please upload a clear hair/scalp photo.");
    if (validation.length) {
      setErrors(validation);
      return;
    }
    setBusy("analysis");
    setErrors([]);
    const form = new FormData();
    form.append("profile_json", JSON.stringify(profile));
    form.append("image", image);
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const payload = await response.json();
    setBusy("");
    if (!response.ok) {
      setErrors(payload.errors || ["Analysis failed. Please try again."]);
      return;
    }
    setResult(payload.result);
    setLabResult(null);
    setChat([]);
  };

  const reviewReport = async () => {
    if (!report || !result) return;
    setBusy("report");
    const form = new FormData();
    form.append("profile_json", JSON.stringify(profile));
    form.append("result_json", JSON.stringify(result));
    form.append("report", report);
    const response = await fetch("/api/lab-report", { method: "POST", body: form });
    const payload = await response.json();
    setBusy("");
    if (response.ok) setLabResult(payload.labResult);
  };

  const sendChat = async () => {
    if (!message.trim() || !result) return;
    const nextChat = [...chat, { role: "user", content: message.trim() }];
    setChat(nextChat);
    setMessage("");
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, result, message, history: nextChat }),
    });
    const payload = await response.json();
    setChat([...nextChat, { role: "assistant", content: payload.answer }]);
  };

  const downloadPdf = async () => {
    const response = await fetch("/api/doctor-summary.pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, result, labResult }),
    });
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${slug(profile.name || "hair")}-doctor-summary.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main>
      <section className="hero">
        <div className="heroCopy">
          <span className="eyebrow">Hair and scalp check</span>
          <h1>Hair Doctor AI</h1>
          <p>Understand possible reasons for hair fall, prepare the right doctor questions, upload reports later, and keep every recommendation safely clinician-guided.</p>
          <div className="heroActions">
            <a href="#check" className="primaryAction">Start check</a>
            <a href="#reports" className="secondaryAction">Upload reports</a>
          </div>
        </div>
        <div className="trustPanel">
          <ShieldCheck />
          <strong>No medicine advice</strong>
          <span>Possible causes, safe next steps, and doctor-ready summaries.</span>
        </div>
      </section>

      <section className="grid" id="check">
        <Card title="Step 1: Details" icon={<ClipboardList />}>
          <div className="segmented">
            {["Self", "Other"].map((item) => (
              <button key={item} className={profile.subject === item ? "active" : ""} onClick={() => update("subject", item)}>{item}</button>
            ))}
          </div>
          <div className="formGrid">
            <Input label={profile.subject === "Self" ? "Your name" : "Person's name"} value={profile.name} onChange={(v) => update("name", v)} />
            <Input label="Age" type="number" value={profile.age} onChange={(v) => update("age", Number(v))} />
            <Select label="Gender" value={profile.gender} onChange={(v) => update("gender", v)} options={["Female", "Male", "Non-binary", "Prefer not to say"]} />
            <Select label="How long?" value={profile.duration} onChange={(v) => update("duration", v)} options={["Less than 1 month", "1 to 3 months", "3 to 6 months", "More than 6 months", "More than 1 year", "More than 3 years", "More than 5 years", "More than 10 years"]} />
            <Select label="Main pattern" value={profile.pattern} onChange={(v) => update("pattern", v)} options={["Overall shedding", "Hairline / temples", "Crown thinning", "Part line widening", "Patchy spots", "Sudden heavy shedding", "Breakage"]} />
            <Select label="Dandruff/flakes" value={profile.dandruff} onChange={(v) => update("dandruff", v)} options={["None", "Mild", "Moderate", "Severe"]} />
            <Select label="Scalp symptoms" value={profile.scalp_symptoms} onChange={(v) => update("scalp_symptoms", v)} options={["None", "Itching", "Pain or tenderness", "Redness/burning", "Oily scalp", "Pus/crusting"]} />
            <Select label="Diet quality" value={profile.diet_quality} onChange={(v) => update("diet_quality", v)} options={["Good", "Average", "Poor"]} />
            <Select label="Family history" value={profile.family_history} onChange={(v) => update("family_history", v)} options={["No", "Yes", "Not sure"]} />
            <Select label="Heat/color styling" value={profile.chemical_treatments} onChange={(v) => update("chemical_treatments", v)} options={["No", "Sometimes", "Often"]} />
            <Select label="Recent illness/stress" value={profile.recent_illness} onChange={(v) => update("recent_illness", v)} options={["No", "Yes", "Not sure"]} />
            <Input label="Sleep hours" type="number" value={profile.sleep_hours} onChange={(v) => update("sleep_hours", Number(v))} />
          </div>
          <Range label="Hair fall level" value={profile.hair_fall_level} onChange={(v) => update("hair_fall_level", Number(v))} />
          <Range label="Stress" value={profile.stress} onChange={(v) => update("stress", Number(v))} />
          <Input label="Current medicines or supplements" value={profile.medications} onChange={(v) => update("medications", v)} />
          <label className="field full">
            <span>Anything else?</span>
            <textarea value={profile.notes} onChange={(e) => update("notes", e.target.value)} placeholder="Postpartum, thyroid, weight loss, new product, diet change..." />
          </label>
        </Card>

        <Card title="Photo" icon={<Camera />}>
          <label className="drop">
            <Upload />
            <strong>Upload clear scalp/hair photo</strong>
            <span>Bright light, steady camera, scalp area visible.</span>
            <input type="file" accept="image/png,image/jpeg" onChange={(e) => onImage(e.target.files?.[0])} />
          </label>
          {preview && <img className="preview" src={preview} alt="Selected hair/scalp" />}
          {errors.length > 0 && <div className="errors">{errors.map((e) => <p key={e}><AlertTriangle size={16} />{e}</p>)}</div>}
          <button className="analyse" onClick={analyze} disabled={busy === "analysis"}>
            <HeartPulse /> {busy === "analysis" ? "Checking..." : "Analyse"}
          </button>
        </Card>
      </section>

      {result && (
        <>
          <section className="resultBand">
            <div className={`score ${result.risk_level.toLowerCase()}`}>
              <span>{firstName}'s concern</span>
              <strong>{result.risk_level}</strong>
              <small>{result.risk_score}/100 concern score</small>
            </div>
            <div>
              <span className="eyebrow dark">What this may mean</span>
              <h2>{firstName}, {lowerFirst(result.summary)}</h2>
              <p>This is informational screening only. Please consult a dermatologist or qualified clinician before using medicines or supplements.</p>
            </div>
          </section>

          <section className="three">
            <InfoList title="Possible causes" items={result.likely_causes} />
            <InfoList title="What to do next" items={result.needed_actions} />
            <InfoList title="Follow-up questions" items={result.follow_up_questions} />
          </section>

          <section className="grid" id="reports">
            <Card title="Step 2: Tests to discuss" icon={<FlaskConical />}>
              <p className="muted">Show this checklist to a dermatologist or qualified clinician. After tests are done, return here and upload the report.</p>
              <InfoList title="Suggested tests" items={recommendedTests} compact />
            </Card>
            <Card title="Step 3: Upload lab reports here" icon={<FileText />}>
              <label className="drop compactDrop">
                <Upload />
                <strong>Upload completed report</strong>
                <span>PDF, JPG, PNG, or TXT.</span>
                <input type="file" accept=".pdf,.txt,image/png,image/jpeg" onChange={(e) => setReport(e.target.files?.[0])} />
              </label>
              {report && <p className="selectedFile">{report.name}</p>}
              <button className="analyse" onClick={reviewReport} disabled={!report || busy === "report"}>
                <Activity /> {busy === "report" ? "Reviewing..." : "Review Test Report"}
              </button>
            </Card>
          </section>

          {labResult && (
            <section className="three">
              <InfoList title="Possible report findings" items={labResult.possible_findings} />
              <InfoList title="Ask your doctor" items={labResult.doctor_discussion} />
              <InfoList title="Still missing" items={labResult.missing_tests?.length ? labResult.missing_tests : ["No key missing tests detected from the readable report."]} />
            </section>
          )}

          <section className="summaryStrip">
            <div>
              <span className="eyebrow dark">Step 4</span>
              <h2>Doctor visit summary</h2>
              <p>Download a clean PDF with symptoms, possible contributors, test checklist, report notes, and safety wording.</p>
            </div>
            <button className="analyse inline" onClick={downloadPdf}><Download /> Download PDF</button>
          </section>

          <section className="chat">
            <div className="chatHeader">
              <MessageCircle />
              <div>
                <h2>Ask follow-up questions for {firstName}</h2>
                <p>Ask about routine, food, dandruff, regrowth time, warning signs, reports, or doctor visit prep.</p>
              </div>
            </div>
            <div className="messages">
              {chat.map((item, index) => <div key={index} className={`bubble ${item.role}`}>{item.content}</div>)}
            </div>
            <div className="chatInput">
              <input value={message} onChange={(e) => setMessage(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendChat()} placeholder="Ask a follow-up question..." />
              <button onClick={sendChat}>Send</button>
            </div>
          </section>
        </>
      )}
    </main>
  );
}

function Card({ title, icon, children }) {
  return <section className="card"><div className="cardTitle">{icon}<h2>{title}</h2></div>{children}</section>;
}

function Input({ label, value, onChange, type = "text" }) {
  return <label className="field"><span>{label}</span><input type={type} value={value} onChange={(e) => onChange(e.target.value)} /></label>;
}

function Select({ label, value, onChange, options }) {
  return <label className="field"><span>{label}</span><select value={value} onChange={(e) => onChange(e.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></label>;
}

function Range({ label, value, onChange }) {
  return <label className="range"><span>{label}: <strong>{value}</strong></span><input type="range" min="0" max="10" value={value} onChange={(e) => onChange(e.target.value)} /></label>;
}

function InfoList({ title, items, compact = false }) {
  return <section className={`infoList ${compact ? "compact" : ""}`}><h3>{title}</h3><ul>{(items || []).map((item) => <li key={item}>{item}</li>)}</ul></section>;
}

function lowerFirst(value = "") {
  return value ? value.charAt(0).toLowerCase() + value.slice(1) : "this screening is complete.";
}

function slug(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "hair-summary";
}

createRoot(document.getElementById("root")).render(<App />);
