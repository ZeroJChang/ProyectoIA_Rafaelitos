import React, { useState } from "react";
import "./App.css";
import groupImage from "./assets/group-photo.png"; // asegúrate de colocar tu imagen en src/assets/

function App() {
  const [news, setNews] = useState("");

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
    if (data.category) {
      alert(`Category predicted: ${data.category}`);
    } else {
      alert("Error: " + data.error);
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Failed to connect to the backend.");
  }
};


  return (
    <div className="container">
	<h1>PROYECT RAFAELITOS</h1>
      <div className="header">
        <img src={groupImage} alt="Group" className="group-photo" />
        <p className="description">
          This project helps us understand how Naïve Bayes works and shows an example of how it can help classify news.
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
    </div>
  );
}

export default App;
