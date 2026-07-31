function renderResults(results) {
  for (let i = 0; i < 6; i++) {
    const img = document.getElementById(`result-${i}`);
    if (results[i]) {
      img.src = results[i].image_url;
      img.onclick = () => openModal(results[i].image_url);
    } else {
      img.src = "";
      img.onclick = null;
    }
  }
}

function showTiming(seconds) {
  document.getElementById("timing").textContent = `${seconds.toFixed(2)}s`;
}

function openModal(src) {
  document.getElementById("modalImg").src = src;
  document.getElementById("modal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}

document.getElementById("modal").addEventListener("click", closeModal);
document.getElementById("modalImg").addEventListener("click", (e) => e.stopPropagation());
document.getElementById("modal-close-btn").addEventListener("click", closeModal);
document.getElementById("search-btn").addEventListener("click", search);
document.getElementById("image-search-btn").addEventListener("click", () => {
  document.getElementById("imageFile").click();
});
document.getElementById("imageFile").addEventListener("change", searchByImage);

async function search() {
  const start = performance.now();
  const q = document.getElementById("q").value;
  const res = await fetch(`http://localhost:8000/search-image?q=${encodeURIComponent(q)}`);
  const data = await res.json();
  renderResults(data.results);
  showTiming((performance.now() - start) / 1000);
}

document.getElementById("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") search();
});

async function searchByImage() {
  const fileInput = document.getElementById("imageFile");
  if (!fileInput.files.length) return;

  const start = performance.now();
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("http://localhost:8000/search-by-image", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  renderResults(data.results);
  showTiming((performance.now() - start) / 1000);
}
