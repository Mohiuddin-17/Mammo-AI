import { useState, useCallback } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Colour map for results
const RESULT_STYLES = {
  Malignant: {
    border: "border-red-500",
    bg: "bg-red-50",
    text: "text-red-700",
    bar: "bg-red-500",
    icon: "⚠️",
  },
  "Benign/Normal": {
    border: "border-emerald-500",
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    bar: "bg-emerald-500",
    icon: "✅",
  },
};

function ConfidenceBar({ label, value, colorClass }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        <span className="text-sm text-gray-500">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${colorClass}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);

    // Preview: skip for DICOM (binary format, can't display natively in browser)
    const ext = f.name.toLowerCase().split(".").pop();
    if (["png", "jpg", "jpeg"].includes(ext)) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null); // DICOM — show placeholder
    }
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const onInputChange = (e) => handleFile(e.target.files[0]);

  const runPrediction = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000,
      });
      setResult(res.data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        "Connection failed. Ensure the API is running.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const style = result ? RESULT_STYLES[result.classification] : null;

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
          <span className="text-white font-bold text-sm">M</span>
        </div>
        <div>
          <h1 className="text-slate-900 font-semibold text-lg leading-none">MammoAI</h1>
          <p className="text-slate-500 text-xs mt-0.5">
            Breast Cancer Screening Assistant — Research Use Only
          </p>
        </div>
        <div className="ml-auto text-xs text-slate-400 bg-amber-50 border border-amber-200 px-3 py-1 rounded-full">
          ⚠️ Not a medical device
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 py-10">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-slate-800 mb-2">
            Mammogram Classification
          </h2>
          <p className="text-slate-500 text-sm max-w-md mx-auto">
            Upload a mammogram image (DICOM, PNG, or JPEG). The AI model will
            classify it as <strong>Malignant</strong> or{" "}
            <strong>Benign/Normal</strong> with a confidence score.
          </p>
        </div>

        {/* Upload Zone */}
        <div
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all
            ${dragOver ? "border-blue-400 bg-blue-50" : "border-slate-300 bg-white hover:border-blue-300 hover:bg-blue-50/30"}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <input
            id="fileInput"
            type="file"
            className="hidden"
            accept=".dcm,.dicom,.png,.jpg,.jpeg"
            onChange={onInputChange}
          />

          {preview ? (
            <img
              src={preview}
              alt="Mammogram preview"
              className="max-h-64 mx-auto rounded-lg object-contain mb-4 shadow"
            />
          ) : file ? (
            <div className="text-5xl mb-3">🩻</div>
          ) : (
            <div className="text-5xl mb-3">📁</div>
          )}

          <p className="text-slate-600 text-sm">
            {file
              ? file.name
              : "Drag & drop your mammogram here, or click to browse"}
          </p>
          {!file && (
            <p className="text-slate-400 text-xs mt-1">
              Supported: .dcm, .dicom, .png, .jpg, .jpeg (max 50 MB)
            </p>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 mt-4">
          <button
            onClick={runPrediction}
            disabled={!file || loading}
            className={`flex-1 py-3 px-6 rounded-xl font-semibold text-sm transition-all
              ${file && !loading
                ? "bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg"
                : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10"
                    stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor"
                    d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Analysing…
              </span>
            ) : (
              "Run Classification"
            )}
          </button>
          {file && (
            <button
              onClick={reset}
              className="py-3 px-5 rounded-xl border border-slate-300 text-slate-600 text-sm hover:bg-slate-100 transition"
            >
              Clear
            </button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mt-5 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Results */}
        {result && style && (
          <div className={`mt-6 rounded-2xl border-2 p-6 ${style.border} ${style.bg}`}>
            <div className="flex items-center gap-3 mb-5">
              <span className="text-3xl">{style.icon}</span>
              <div>
                <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Classification Result
                </p>
                <p className={`text-2xl font-bold ${style.text}`}>
                  {result.classification}
                </p>
              </div>
            </div>

            <div className="bg-white/60 rounded-xl p-4">
              <p className="text-xs font-semibold text-slate-500 uppercase mb-3 tracking-wide">
                Probability Distribution
              </p>
              <ConfidenceBar
                label="Malignant"
                value={result.malignant_probability}
                colorClass="bg-red-400"
              />
              <ConfidenceBar
                label="Benign / Normal"
                value={result.benign_probability}
                colorClass="bg-emerald-400"
              />
            </div>

            <div className="mt-4 bg-amber-50 border border-amber-200 rounded-xl p-3 text-xs text-amber-800">
              <strong>Clinical Disclaimer: </strong>{result.warning}
            </div>
          </div>
        )}

        {/* Footer note */}
        <p className="text-center text-xs text-slate-400 mt-10">
          Model trained on CBIS-DDSM · EfficientNet-B4 · For research use only
        </p>
      </main>
    </div>
  );
}