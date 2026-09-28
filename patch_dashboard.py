import sys

file_path = "/home/ai-mini-playback/Main Server/cctv-surveillance/frontend/src/pages/Dashboard.tsx"
with open(file_path, "r") as f:
    content = f.read()

state_declarations = """  const [theftTargetCams, setTheftTargetCams] = useState<number[]>([]);
  const [showTheftModal, setShowTheftModal] = useState(false);

  useEffect(() => {
    fetch(`${API_V1_URL}/motion/theft-cams`)
      .then(res => res.json())
      .then(data => setTheftTargetCams(data.cameras || []))
      .catch(e => console.error(e));
  }, []);

  const saveTheftCams = (cams: number[]) => {
    setTheftTargetCams(cams);
    fetch(`${API_V1_URL}/motion/theft-cams`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cameras: cams })
    }).catch(e => console.error(e));
  };
"""

# Insert state
if "theftTargetCams" not in content:
    content = content.replace("const [motionTargetCams, setMotionTargetCams] = useState<number[]>([]);", 
                             "const [motionTargetCams, setMotionTargetCams] = useState<number[]>([]);\n" + state_declarations)

# Find the place to insert the button
# In the motion header, next to the Enable Motion Detection button
# line 1218: : "Enable motion detection"
#   }
# </button>

button_code = """
              <button 
                onClick={() => setShowTheftModal(true)}
                className={`db-page-btn db-grid-switcher__btn db-motion-toggle-btn`}
                style={{ marginLeft: '10px', background: theftTargetCams.length > 0 ? 'var(--blue)' : 'var(--glass-bg)', color: 'white', border: '1px solid var(--glass-border)' }}
              >
                ☁️ Cloud Theft AI ({theftTargetCams.length})
              </button>
"""
if "☁️ Cloud Theft AI" not in content:
    content = content.replace(': "Enable motion detection"\n                }\n              </button>', 
                              ': "Enable motion detection"\n                }\n              </button>\n' + button_code)

# Add Modal at the end of the file before last </div>
modal_code = """
      {/* Cloud Theft AI Modal */}
      {showTheftModal && (
        <div className="db-modal-overlay">
          <div className="db-modal">
            <h2>Select Cameras for Cloud Theft AI</h2>
            <p>When motion is detected on these cameras locally, a 5s clip will be sent to AWS for Theft Analysis.</p>
            <div className="db-modal-content" style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {cameras.map(c => (
                <label key={c.id} style={{ display: 'flex', alignItems: 'center', margin: '10px 0', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={theftTargetCams.includes(c.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        saveTheftCams([...theftTargetCams, c.id]);
                      } else {
                        saveTheftCams(theftTargetCams.filter(id => id !== c.id));
                      }
                    }}
                    style={{ marginRight: '10px', width: '18px', height: '18px' }}
                  />
                  {c.name}
                </label>
              ))}
            </div>
            <div className="db-modal-actions">
              <button className="db-btn db-btn-primary" onClick={() => setShowTheftModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
"""

if "Cloud Theft AI Modal" not in content:
    content = content.replace("export default Dashboard;", modal_code + "\nexport default Dashboard;")

with open(file_path, "w") as f:
    f.write(content)
print("Dashboard.tsx patched successfully!")
