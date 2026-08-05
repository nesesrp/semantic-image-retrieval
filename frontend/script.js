const API_BASE = "http://localhost:8000";

function renderResults(results) {
  for (let i = 0; i < 6; i++) {
    const img = document.getElementById(`result-${i}`);
    const score = document.getElementById(`score-${i}`);

    if (results[i]) {
      img.onclick = () => openModal(results[i].image_url, results[i].filename);
      img.src = results[i].image_url;
      score.textContent = `Score: ${results[i].score.toFixed(4)}`;
    } else {
      img.onclick = null;
      img.src = "";
      score.textContent = "";
    }
  }
}

function showTiming(seconds) {
  document.getElementById("timing").textContent = `${seconds.toFixed(2)}s`;
}

async function openModal(src, filename) {
  document.getElementById("modalImg").src = src;
  document.getElementById("modal").dataset.filename = filename;
  document.getElementById("modal").classList.remove("hidden");

  document.getElementById("modalQuestion").value = "";
  document.getElementById("modalAnswer").textContent = "";

  const captionEl = document.getElementById("modalCaption");
  captionEl.textContent = "Describing image...";
  try {
    const res = await fetch(`${API_BASE}/describe/${encodeURIComponent(filename)}`);
    const data = await res.json();
    captionEl.textContent = data.caption;
  } catch (err) {
    captionEl.textContent = "Could not describe image.";
  }
}

function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}

async function askModalQuestion() {
  const filename = document.getElementById("modal").dataset.filename;
  const question = document.getElementById("modalQuestion").value.trim();
  if (!filename || !question) return;

  const answerEl = document.getElementById("modalAnswer");
  answerEl.textContent = "Thinking...";
  try {
    const res = await fetch(
      `${API_BASE}/ask/${encodeURIComponent(filename)}?q=${encodeURIComponent(question)}`
    );
    const data = await res.json();
    answerEl.textContent = data.answer;
  } catch (err) {
    answerEl.textContent = "Could not get an answer.";
  }
}

function downloadModalImage() {
  const filename = document.getElementById("modal").dataset.filename;
  if (!filename) return;
  const a = document.createElement("a");
  a.href = `${API_BASE}/download/${filename}`;
  a.click();
}

document.getElementById("modal").addEventListener("click", closeModal);
document.getElementById("modalContent").addEventListener("click", (e) => e.stopPropagation());
document.getElementById("modal-close-btn").addEventListener("click", closeModal);
document.getElementById("modal-download-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  downloadModalImage();
});
document.getElementById("modal-ask-btn").addEventListener("click", askModalQuestion);
document.getElementById("modalQuestion").addEventListener("keydown", (e) => {
  if (e.key === "Enter") askModalQuestion();
});
document.getElementById("search-btn").addEventListener("click", search);
document.getElementById("image-search-btn").addEventListener("click", () => {
  document.getElementById("imageFile").click();
});
document.getElementById("imageFile").addEventListener("change", searchByImage);

async function search() {
  const start = performance.now();
  const q = document.getElementById("q").value;
  const res = await fetch(`${API_BASE}/search-image?q=${encodeURIComponent(q)}`);
  const data = await res.json();
  renderResults(data.results);
  showTiming((performance.now() - start) / 1000);
}

document.getElementById("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") search();
});


const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
async function searchByImage() {
  const fileInput = document.getElementById("imageFile");
  if (!fileInput.files.length) return;

  const file = fileInput.files[0];
  if (file.size > MAX_FILE_SIZE) {
    alert("Image must be smaller than 10 MB.");
    fileInput.value = "";
    return;
  }

  const start = performance.now();
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/search-by-image`, {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  renderResults(data.results);
  showTiming((performance.now() - start) / 1000);
}
