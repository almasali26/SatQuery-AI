import { useEffect, useRef, useState } from 'react'
import { analyzeImage } from './services/api.js'
import './App.css'

const acceptedFormats = '.jpg,.jpeg,.png,.tif,.tiff'
const BACKEND_URL = 'https://satquery-ai-a20n.onrender.com'

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function UploadArea({ label, image, onSelect, onRemove, compact = false }) {
  return (
    <div className={`upload-area ${compact ? 'upload-area--compact' : ''}`}>
      <input
        type="file"
        accept={acceptedFormats}
        onChange={(event) =>
          onSelect(event.target.files?.[0] ?? null)
        }
        aria-label={`Upload ${label}`}
      />

      {image ? (
        <img
          className="preview-image"
          src={image.url}
          alt={`${label} preview`}
        />
      ) : (
        <div className="upload-prompt">
          <span className="upload-icon" aria-hidden="true">↥</span>
          <strong>Drop an image here</strong>
          <span>or click to browse</span>
          <small>JPG, PNG, TIFF up to 25 MB</small>
        </div>
      )}

      {image && (
        <div className="image-meta">
          <span className="image-name">{image.name}</span>
          <span className="image-size">
            {formatFileSize(image.size)}
          </span>
          <button
            className="remove-button"
            type="button"
            onClick={onRemove}
          >
            Remove Image
          </button>
        </div>
      )}
    </div>
  )
}

/* Visual highlighting component */
function HighlightedImage({ image, regions }) {
  if (!image) return null

  const visibleRegions = (regions || []).filter(
    (region) => region.bounding_box
  )

  if (visibleRegions.length === 0) {
    return null
  }

  return (
    <div className="highlight-section">
      <div className="highlight-title">
        <div>
          <span className="result-label">VISUAL ANALYSIS</span>
          <h3>Detected regions</h3>
        </div>
        <span className="region-count">
          {visibleRegions.length} region
          {visibleRegions.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="highlight-viewer">
        <img
          src={image.url}
          alt="Analyzed satellite scene with highlighted regions"
        />

        {visibleRegions.map((region, index) => {
          const box = region.bounding_box

          return (
            <div
              key={`${region.feature}-${index}`}
              className="highlight-box"
              style={{
                left: `${box.x_min}px`,
                top: `${box.y_min}px`,
                width: `${Math.max(1, box.x_max - box.x_min)}px`,
                height: `${Math.max(1, box.y_max - box.y_min)}px`,
              }}
              title={`${region.feature} - ${region.confidence}% confidence`}
            >
              <span className="highlight-label">
                {region.feature}
              </span>
            </div>
          )
        })}
      </div>

      <div className="region-list">
        {visibleRegions.map((region, index) => (
          <div
            className="region-item"
            key={`${region.feature}-info-${index}`}
          >
            <span className="region-dot" />
            <div>
              <strong>
                {region.feature.charAt(0).toUpperCase() +
                  region.feature.slice(1)}
              </strong>
              <span>
                {region.area_percent}% of image ·{' '}
                {region.confidence}% confidence
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function App() {
  const [mainImage, setMainImage] = useState(null)
  const [compareImages, setCompareImages] = useState([null, null])
  const [question, setQuestion] = useState('')
  const [compareQuestion, setCompareQuestion] = useState(
  'Analyze the optical and SAR information together.'
)
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef(null)
  const [hasAnalyzed, setHasAnalyzed] = useState(false)

  const [analysisResult, setAnalysisResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState('')
  const [isComparing, setIsComparing] = useState(false)
  const [comparisonResult, setComparisonResult] = useState(null)
  const [comparisonError, setComparisonError] = useState('')

  const mainImageRef = useRef(null)
  const compareImagesRef = useRef(compareImages)

  useEffect(() => {
    mainImageRef.current = mainImage
    compareImagesRef.current = compareImages
  }, [mainImage, compareImages])

  useEffect(() => () => {
    if (mainImageRef.current) {
      URL.revokeObjectURL(mainImageRef.current.url)
    }

    compareImagesRef.current.forEach((image) => {
      if (image) URL.revokeObjectURL(image.url)
    })
  }, [])

  const selectMainImage = (file) => {
    setHasAnalyzed(false)
    setAnalysisResult(null)
    setAnalysisError('')

    setMainImage((current) => {
      if (current) {
        URL.revokeObjectURL(current.url)
      }

      return file
        ? {
            file,
            name: file.name,
            size: file.size,
            url: URL.createObjectURL(file),
          }
        : null
    })
  }

  const selectCompareImage = (index, file) => {
    setComparisonResult(null)
    setComparisonError('')

    setCompareImages((current) => {
      const next = [...current]

      if (next[index]) {
        URL.revokeObjectURL(next[index].url)
      }

      next[index] = file
        ? {
            file,
            name: file.name,
            size: file.size,
            url: URL.createObjectURL(file),
          }
        : null

      return next
    })
  }

  const handleCompare = async () => {
  const [image1, image2] = compareImages

  if (!image1 || !image2) return

  const formData = new FormData()
  formData.append('image1', image1.file)
  formData.append('image2', image2.file)
  formData.append(
    'question',
    compareQuestion.trim() || 'Compare these two images and identify the changes.'
  )

  setIsComparing(true)
  setComparisonResult(null)
  setComparisonError('')

  try {
    const response = await fetch(`${BACKEND_URL}/compare`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error('The backend could not compare these images.')
    }

    const result = await response.json()

    setComparisonResult(result)
  } catch {
    setComparisonError(
      'We could not reach the change detection service. Please make sure the backend is running and try again.'
    )
  } finally {
    setIsComparing(false)
  }
}
const handleVoiceInput = () => {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition

  if (!SpeechRecognition) {
    alert('Voice recognition is not supported in this browser.')
    return
  }

  if (isListening) {
    recognitionRef.current?.stop()
    setIsListening(false)
    return
  }

  const recognition = new SpeechRecognition()

  recognition.lang = 'en-US'
  recognition.continuous = false
  recognition.interimResults = false

  recognition.onstart = () => {
    setIsListening(true)
  }

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript
    setQuestion(transcript)
  }

  recognition.onerror = () => {
    setIsListening(false)
  }

  recognition.onend = () => {
    setIsListening(false)
  }

  recognitionRef.current = recognition
  recognition.start()
}


  const handleAnalyze = async () => {
    if (!mainImage) return

    setIsAnalyzing(true)
    setHasAnalyzed(false)
    setAnalysisError('')
    setAnalysisResult(null)

    try {
      const result = await analyzeImage(mainImage, question)

      setAnalysisResult(result)
      setHasAnalyzed(true)
    } catch {
      setAnalysisError(
        'We could not reach the analysis service. Please make sure the backend is running and try again.'
      )
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="app-shell">

      <header className="topbar">
        <a
          className="brand"
          href="/"
          aria-label="SatQuery AI home"
        >
          <span className="brand-mark" aria-hidden="true">✦</span>
          <span>
            SatQuery <b>AI</b>
          </span>
        </a>

        <span className="status">
          <i />
          Demo workspace
        </span>
      </header>

      <main className="dashboard">

        <section className="intro">
          <div>
            <p className="eyebrow">
              SATELLITE INTELLIGENCE PLATFORM
            </p>

            <h1>
              Understand satellite images
              <br />
              <em>using natural language.</em>
            </h1>

            <p className="intro-copy">
              Upload a scene, ask a question, and turn complex
              earth observation data into a clear answer.
            </p>
          </div>

          <div className="orbit-art" aria-hidden="true">
            <span className="orbit orbit-one" />
            <span className="orbit orbit-two" />
            <span className="planet" />
            <span className="signal signal-one" />
            <span className="signal signal-two" />
          </div>
        </section>

        <section className="workspace-grid">

          <div className="analysis-column">

            <div className="section-heading">
              <div>
                <span className="step">01</span>

                <div>
                  <h2>Analyze an image</h2>
                  <p>
                    Ask questions about any satellite scene
                  </p>
                </div>
              </div>

              <span className="live-tag">READY</span>
            </div>

            <div className="panel analysis-panel">

              <UploadArea
                label="satellite image"
                image={mainImage}
                onSelect={selectMainImage}
                onRemove={() => selectMainImage(null)}
              />

              <div className="question-block">
                <label htmlFor="question">
                  What would you like to know?
                </label>

                <div className="question-input">
                  <span aria-hidden="true">⌕</span>

                  <input
                    id="question"
                    value={question}
                    onChange={(event) =>
                      setQuestion(event.target.value)
                    }
                    placeholder="Ask something about this satellite image..."
                  />
                  <button
                    type="button"
                    className="voice-input-button"
                    onClick={handleVoiceInput}
                    aria-label={isListening ? "Stop listening" : "Start listening"}
                  >
                    {isListening ? (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="6" y="3" width="12" height="14" rx="6" />
    <path d="M4 11a8 8 0 0 0 16 0" />
    <line x1="12" y1="19" x2="12" y2="22" />
    <line x1="9" y1="22" x2="15" y2="22" />
  </svg>
) : (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M12 2a4 4 0 0 0-4 4v6a4 4 0 0 0 8 0V6a4 4 0 0 0-4-4Z" />
    <path d="M5 10a7 7 0 0 0 14 0" />
    <line x1="12" y1="17" x2="12" y2="22" />
    <line x1="9" y1="22" x2="15" y2="22" />
  </svg>
)}
                  </button>
                </div>
              </div>

              <button
                className="primary-button"
                type="button"
                disabled={!mainImage || isAnalyzing}
                onClick={handleAnalyze}
              >
                <span>
                  {isAnalyzing
                    ? 'Analyzing...'
                    : 'Analyze Image'}
                </span>

                <span aria-hidden="true">
                  {isAnalyzing ? '◌' : '→'}
                </span>
              </button>

              {!mainImage && (
                <p className="helper-text">
                  Upload an image to enable analysis
                </p>
              )}
            </div>

            <div className="section-heading result-heading">
              <div>
                <span className="step">02</span>

                <div>
                  <h2>Analysis result</h2>
                  <p>
                    Your AI-powered insight will appear here
                  </p>
                </div>
              </div>
            </div>

            <div
              className={`panel result-panel ${
                hasAnalyzed
                  ? 'result-panel--active'
                  : ''
              }`}
            >

              {isAnalyzing ? (
                <div className="empty-result">
                  <span className="empty-icon loading-icon">
                    ◌
                  </span>

                  <strong>Analyzing image...</strong>

                  <span>
                    Sending your image and question to the
                    backend
                  </span>
                </div>

              ) : analysisError ? (

                <div className="empty-result error-result">
                  <span className="empty-icon">!</span>

                  <strong>Analysis unavailable</strong>

                  <span>{analysisError}</span>
                </div>

              ) : analysisResult ? (

                <div className="result-content">

                  <span className="result-icon">✦</span>

                  <div>
                    <span className="result-label">
                      A                    I ANSWER
                    </span>

                    <h3>
                      {analysisResult.answer}
                    </h3>

                    <p>
                      {analysisResult.explanation || 'The AI Analysis was completed using the selected image and analysis model.'}
                    </p>
                  </div>

                  <span className="confidence">
                    {analysisResult.confidence > 0 ? `${analysisResult.confidence}% confidence` : 'Confidence score unavailable'}
                  </span>
                  {analysisResult.agent_decision && (
  <div className="agent-execution">
    <div className="agent-header">
      <div className="agent-title">
        <span className="agent-icon">✦</span>
        <div>
          <span className="result-label">AI AGENT</span>
          <strong>Agentic Execution</strong>
        </div>
      </div>

      <span className="agent-status">
        <i />
        COMPLETED
      </span>
    </div>

    <div className="agent-flow">
      <div className="agent-step">
        <span className="agent-step-number">01</span>
        <div>
          <strong>Query understood</strong>
          <span>
            Natural-language question received
          </span>
        </div>
        <b>✓</b>
      </div>

      <div className="agent-line" />

      <div className="agent-step">
        <span className="agent-step-number">02</span>
        <div>
          <strong>Task selected</strong>
          <span>
            {analysisResult.agent_decision.task}
          </span>
        </div>
        <b>✓</b>
      </div>

      <div className="agent-line" />

      <div className="agent-step">
        <span className="agent-step-number">03</span>
        <div>
          <strong>Tool executed</strong>
          <span>
            {analysisResult.agent_decision.tools?.join(', ') || 'N/A'}
          </span>
        </div>
        <b>✓</b>
      </div>
    </div>

    <div className="agent-reason">
      <span>WHY THIS TOOL?</span>
      <p>{analysisResult.agent_decision.reason}</p>
    </div>
  </div>
)}
                </div>

              ) : (

                <div className="empty-result">
                  <span className="empty-icon">✧</span>

                  <strong>No analysis yet</strong>

                  <span>
                    Upload an image and ask a question to
                    get started
                  </span>
                </div>
              )}

              <div className="result-features">
                <span>◉ AI answer</span>
                <span>◌ Confidence score</span>
                <span>▧ Highlighted regions</span>
                <span>≡ Explanation</span>
              </div>

              {analysisResult && (
                <div className="segmentation-overlay-section">
                  <span className="result-label">AI SEGMENTATION OVERLAY</span>
                  {analysisResult.segmentation_overlay ? (
                    <img
                      className="segmentation-overlay-image"
                      src={`${BACKEND_URL}${analysisResult.segmentation_overlay}`}
                      alt="AI segmentation overlay of the analyzed satellite image"
                    />
                  ) : (
                    <p className="overlay-fallback">
                      No segmentation overlay was returned for this analysis.
                    </p>
                  )}
                </div>
              )}

            </div>

            {/* Visual highlighting */}
            {analysisResult && mainImage && (
              <HighlightedImage
                image={mainImage}
                regions={analysisResult.detected_regions}
              />
            )}

          </div>

          <aside className="compare-card">

            <div className="section-heading compare-heading">

              <div>
                <span className="step">03</span>

                <div>
                  <h2>Compare Two Images</h2>
                  <p>
                    Detect changes over time
                  </p>
                </div>
              </div>

              <span className="live-tag">
                READY
              </span>

            </div>

            <p className="compare-copy">
              Place two satellite images side by side to
              discover meaningful changes in a location.
            </p>

            <div className="compare-uploads">

              {['Image 1', 'Image 2'].map(
                (label, index) => (
                  <div
                    className="compare-item"
                    key={label}
                  >
                    <label>{label}</label>

                    <UploadArea
                      label={label}
                      image={compareImages[index]}
                      onSelect={(file) =>
                        selectCompareImage(index, file)
                      }
                      onRemove={() =>
                        selectCompareImage(index, null)
                      }
                      compact
                    />
                  </div>
                )
              )}

            </div>

            <div className="compare-question">
  <label htmlFor="compare-question">
    Comparison Question
  </label>

  <textarea
    id="compare-question"
    value={compareQuestion}
    onChange={(event) => setCompareQuestion(event.target.value)}
    placeholder="Ask what you want to analyze..."
    rows={3}
  />
</div>

            <button
              className="secondary-button"
              type="button"
              disabled={!compareImages[0] || !compareImages[1] || isComparing}
              onClick={handleCompare}
            >
              {isComparing ? 'Comparing...' : 'Compare Images'}
              <span aria-hidden="true">{isComparing ? '◌' : '→'}</span>
            </button>

          </aside>

        </section>

        <section className="change-result-section">
          <div className="section-heading result-heading">
            <div>
              <span className="step">04</span>
              <div>
                <h2>Change Detection Result</h2>
                <p>Compare both scenes to reveal meaningful changes</p>
              </div>
            </div>
          </div>

          <div className="panel result-panel">
            {isComparing ? (
              <div className="empty-result">
                <span className="empty-icon loading-icon">◌</span>
                <strong>Comparing images...</strong>
                <span>Sending both satellite images to the backend</span>
              </div>
            ) : comparisonError ? (
              <div className="empty-result error-result">
                <span className="empty-icon">!</span>
                <strong>Change detection unavailable</strong>
                <span>{comparisonError}</span>
              </div>
            ) : comparisonResult ? (
              <div className="result-content">
                <span className="result-icon">↔</span>
                <div>
                  <span className="result-label">CHANGE ANALYSIS</span>
                  <h3>
                    {comparisonResult.change_percentage ?? comparisonResult.changePercentage ?? 0}% changed
                  </h3>
                  <p>
                    {comparisonResult.explanation || 'The backend returned a change detection result.'}
                  </p>
                  {comparisonResult.change_map && (
  <img
    className="change-map-preview"
    src={`${BACKEND_URL}${comparisonResult.change_map}`}
    alt="Satellite image change map"
  />
)}

{comparisonResult.agent_decision && (
  <div className="agent-execution">
    <span className="result-label">AGENTIC EXECUTION</span>

    <div className="agent-details">
      <div>
        <strong>Selected Task</strong>
        <span>{comparisonResult.agent_decision.task}</span>
      </div>

      <div>
        <strong>Tool Used</strong>
        <span>
          {comparisonResult.agent_decision.tools?.join(', ') || 'N/A'}
        </span>
      </div>

      <div>
        <strong>Reason</strong>
        <span>{comparisonResult.agent_decision.reason}</span>
      </div>
    </div>
  </div>
)}
                </div>
              </div>
            ) : (
              <div className="empty-result">
                <span className="empty-icon">↔</span>
                <strong>No comparison yet</strong>
                <span>Select Image 1 and Image 2 to detect changes</span>
              </div>
            )}

            <div className="result-features">
              <span>◌ Change percentage</span>
              <span>≡ Explanation</span>
              <span>▧ Change map</span>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <span>
          SIH26167 · Satellite intelligence for everyone
        </span>

        <span>
          Built for exploration
          <span className="footer-star">✦</span>
        </span>
      </footer>

    </div>
  )
}

export default App