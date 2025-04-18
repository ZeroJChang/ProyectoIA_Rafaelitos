import React, { useState } from "react";
import "./App.css";
import groupImage from "./assets/group-photo.png"; 

function App() {
  const [news, setNews] = useState("");
  const [results, setResults] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const handleVerify = async () => {
    if (!news.trim()) {
      alert("Please enter some news text.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: news }),
      });

      const data = await response.json();
      setResults(data);
      setShowModal(true);
    } catch (error) {
      console.error("Error:", error);
      setResults({ error: "Failed to connect to backend." });
      setShowModal(true);
    }
  };

  const closeModal = () => {
    setShowModal(false);
  };

  return (
    <div className="container">
      <h1>PROYECT RAFAELITOS</h1>
      <div className="header">
        <img src={groupImage} alt="Group" className="group-photo" />
        <p className="description">
          This project helps us understand how Naïve Bayes works and shows an
          example of how it can help classify news.
        </p>
      </div>
      <label htmlFor="newsInput">Insert News</label>
      <textarea
        id="newsInput"
        rows="6"
        value={news}
        onChange={(e) => setNews(e.target.value)}
        placeholder="Write your news here..."
      />
      <button onClick={handleVerify}>Verify</button>

      {showModal && results && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Classification Results</h2>
            {results.error && <p>{results.error}</p>}
            {results.categories && (
				  <ul>
					<li style={{ fontWeight: "bold", color: "#28a745", fontSize: "18px" }}>
					  ✅ {results.categories[0].category.toUpperCase()} — {results.categories[0].confidence}%
					</li>
					{results.categories[1] && results.categories[1].confidence > 0 && (
					  <li style={{ color: "#333", marginTop: "10px" }}>
						{results.categories[1].category} — {results.categories[1].confidence}%
					  </li>
					)}
				  </ul>
				)}
            <button className="close-button" onClick={closeModal}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
