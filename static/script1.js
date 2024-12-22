function showContactForm() {
    document.getElementById('contactForm').style.display = 'block';
    document.getElementById('registerForm').style.display = 'none';
}

function showRegisterForm() {
    document.getElementById('contactForm').style.display = 'none';
    document.getElementById('registerForm').style.display = 'block';
}








const bgAnimation = document.getElementById('bgAnimation');
const numberOfColorBoxes = 400;

for (let i = 0; i < numberOfColorBoxes; i++) {
    const colorbox = document.createElement('div');
    colorbox.classList.add('colorBox');
    bgAnimation.append(colorBox)

}

function showHome() {
    document.getElementById('home').style.display = 'block';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'none';
    document.getElementById('register').style.display = 'none';
}

function showTeam() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'block';
    document.getElementById('about').style.display = 'none';
    document.getElementById('register').style.display = 'none';
}

function showAbout() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'block';
    document.getElementById('register').style.display = 'none';
}

function showRegister() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'none';
    document.getElementById('register').style.display = 'block';

}

// Handle chat input
// function handleChat(event) {
//     event.preventDefault(); // Prevent form from submitting

//     const userInput = document.getElementById('user-input').value;
//     const chatbox = document.getElementById('chatbox');

//     if (userInput.trim() === '') return; // Ignore empty input

//     // Create user chat bubble
//     const userBubble = document.createElement('div');
//     userBubble.classList.add('chat-bubble', 'user-bubble');
//     userBubble.innerText = userInput;
//     chatbox.appendChild(userBubble);

//     // Scroll chatbox to the bottom
//     chatbox.scrollTop = chatbox.scrollHeight;

//     // Clear input field
//     document.getElementById('user-input').value = '';



async function handleChat(event) {
    event.preventDefault(); // Prevent form from refreshing the page

    const userInput = document.getElementById('user-input').value;

    if (userInput.trim() === "") {
        alert("Please enter a question!");
        return;
    }

    // Send a POST request to the Flask backend
    const response = await fetch('/predict1', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userInput }) // Match the key used in Flask
    });

    if (!response.ok) {
        const errorData = await response.json();
        alert(errorData.error || "Something went wrong!");
        return;
    }

    const data = await response.json();

    // Display AI assistant's response in the chatbox
    const chatbox = document.getElementById('chatbox');

    // Create user message
    const userMessage = document.createElement('p');
    userMessage.innerHTML = `<strong>You:</strong> ${userInput}`;
    chatbox.appendChild(userMessage);

    // Create AI response message
    const aiResponse = document.createElement('p');
    aiResponse.innerHTML = `<strong>AI Assistant:</strong> ${data.response.result || data.response.error || "No response received"}`;
    chatbox.appendChild(aiResponse);

    // Scroll chatbox to the bottom
    chatbox.scrollTop = chatbox.scrollHeight;

    // Clear the input
    document.getElementById('user-input').value = '';
}

// Function to generate greeting message
function greetUser() {
    const hours = new Date().getHours();
    let greetingMessage = '';

    // Determine the time of day for personalized greeting
    if (hours >= 5 && hours < 12) {
        greetingMessage = "Good Morning! Welcome to our AI-Powered T-Shirt Store! 🌞";
    } else if (hours >= 12 && hours < 17) {
        greetingMessage = "Good Afternoon! Explore the latest T-shirts with AI assistance! ☀";
    } else {
        greetingMessage = "Good Evening! Let's find the perfect T-shirt for you with AI! 🌙";
    }

    // Display greeting message in the greeting container
    document.getElementById('greeting').innerHTML = `
    <h2>${greetingMessage}</h2>
    <p>How can I assist you today?</p>
`;
}

// Call the greetUser function when the page loads
window.onload = greetUser;