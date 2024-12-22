// scripts.js


const bgAnimation = document.getElementById('bgAnimation');
const numberOfColorBoxes = 400;

for (let i = 0; i < numberOfColorBoxes; i++) {
    const colorBox = document.createElement('div');
    colorBox.classList.add('colorBox');
    bgAnimation.append(colorBox);
}

function showHome() {
    document.getElementById('home').style.display = 'block';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'none';
    document.getElementById('contact').style.display = 'none';

    // Show the three dots menu only on the Home page
    document.getElementById('three-dots-menu').style.display = 'block';
}

function showTeam() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'block';
    document.getElementById('about').style.display = 'none';
    document.getElementById('contact').style.display = 'none';

    // Hide the three dots menu on the Team page
    document.getElementById('three-dots-menu').style.display = 'none';
}

function showAbout() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'block';
    document.getElementById('contact').style.display = 'none';

    // Hide the three dots menu on the About page
    document.getElementById('three-dots-menu').style.display = 'none';
}

function contact() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'none';
    document.getElementById('contact').style.display = 'block';

    // Hide the three dots menu on the Contact page
    document.getElementById('three-dots-menu').style.display = 'none';
}


let responseInterval; // To store the interval for response animation
let isStopped = false; // Flag to indicate if the response generation is stopped

// Function to handle chat submission
async function handleChat(event) {
    // Prevent the default form submission behavior
    if (event) event.preventDefault();

    const userInput = document.getElementById('user-input').value.trim();

    // Retrieve the store ID from localStorage
    const storeId = localStorage.getItem('store_id');

    if (userInput === "") {
        alert("Please enter a question!");
        return;
    }

    if (!storeId) {
        alert("Please select a store from the login page!");
        return;
    }

    const chatbox = document.getElementById('chatbox');

    // Display user message
    const userMessage = document.createElement('div');
    userMessage.className = 'chat-message user';
    userMessage.innerHTML = `<p>${userInput}</p>`;
    chatbox.appendChild(userMessage);

    // Add AI response placeholder
    const aiResponsePlaceholder = document.createElement('div');
    aiResponsePlaceholder.className = 'chat-message ai';
    aiResponsePlaceholder.innerHTML = `
        <span class="typing-effect-circle">N</span>
        <span class="typing-effect-text"></span>
    `;
    chatbox.appendChild(aiResponsePlaceholder);

    // Enable the stop button
    const stopButton = document.getElementById('stop-button');
    stopButton.disabled = false;
    isStopped = false;

    const typingCircle = aiResponsePlaceholder.querySelector('.typing-effect-circle');
    const typingText = aiResponsePlaceholder.querySelector('.typing-effect-text');
    const assistantText = "AI Assistant:";
    let currentIndex = 0;

    // Animate "AI Assistant:"
    const animateAssistantText = setInterval(() => {
        if (isStopped) {
            clearInterval(animateAssistantText);
            return;
        }

        if (currentIndex < assistantText.length) {
            typingText.textContent = assistantText.slice(0, currentIndex + 1);
            currentIndex++;
        } else {
            clearInterval(animateAssistantText);
            typingCircle.remove();
            const responseSpan = document.createElement('span');
            responseSpan.className = 'response-text';
            typingText.appendChild(responseSpan);
            fetchAndGenerateResponse(responseSpan, userInput, storeId);
        }
    }, 100);

    // Clear input field
    document.getElementById('user-input').value = '';
}

// Function to fetch and generate AI response
async function fetchAndGenerateResponse(responseSpan, userInput, storeId) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: userInput,
                store_id: storeId
            }),
        });

        const data = await response.json();
        console.log("Backend Response:", data); // Debugging
        let finalResponse = data.response || data.error || "No response received.";

        if (Array.isArray(finalResponse)) {
            finalResponse = finalResponse.join(", ");
        }

        animateResponse(responseSpan, finalResponse);
    } catch (error) {
        responseSpan.textContent = "An unexpected error occurred. Please try again later.";
        document.getElementById('stop-button').disabled = true;
    }
}

// Function to animate the AI response text
function animateResponse(responseSpan, finalResponse) {
    let responseIndex = 0;
    responseSpan.textContent = '';

    responseInterval = setInterval(() => {
        if (isStopped) {
            clearInterval(responseInterval);
            return;
        }

        if (responseIndex < finalResponse.length) {
            responseSpan.textContent += finalResponse[responseIndex];
            responseIndex++;
            document.getElementById('chatbox').scrollTop = document.getElementById('chatbox').scrollHeight;
        } else {
            clearInterval(responseInterval);
            document.getElementById('stop-button').disabled = true;
        }
    }, 50);
}

// Function to stop the AI response animation
function stopResponse() {
    isStopped = true;
    clearInterval(responseInterval);
    document.getElementById('stop-button').disabled = true;
}










// staff

let responseInterval1; // To store the interval for response animation
let isStopped1 = false; // Flag to indicate if the response generation is stopped

// Function to handle chat submission
async function StaffhandleChat(event) {
    // Prevent the default form submission behavior
    if (event) event.preventDefault();

    const userInput = document.getElementById('user-input').value.trim();

    // Retrieve the store ID from localStorage
    const storeId = localStorage.getItem('store_id');

    if (userInput === "") {
        alert("Please enter a question!");
        return;
    }

    if (!storeId) {
        alert("Please select a store from the login page!");
        return;
    }

    const chatbox = document.getElementById('chatbox');

    // Display user message
    const userMessage = document.createElement('div');
    userMessage.className = 'chat-message user';
    userMessage.innerHTML = `<p>${userInput}</p>`;
    chatbox.appendChild(userMessage);

    // Add AI response placeholder
    const aiResponsePlaceholder = document.createElement('div');
    aiResponsePlaceholder.className = 'chat-message ai';
    aiResponsePlaceholder.innerHTML = `
        <span class="typing-effect-circle">N</span>
        <span class="typing-effect-text"></span>
    `;
    chatbox.appendChild(aiResponsePlaceholder);

    // Enable the stop button
    const stopButton = document.getElementById('stop-button');
    stopButton.disabled = false;
    isStopped1 = false;

    const typingCircle = aiResponsePlaceholder.querySelector('.typing-effect-circle');
    const typingText = aiResponsePlaceholder.querySelector('.typing-effect-text');
    const assistantText = "AI Assistant:";
    let currentIndex = 0;

    // Animate "AI Assistant:"
    const animateAssistantText = setInterval(() => {
        if (isStopped1) {
            clearInterval(animateAssistantText);
            return;
        }

        if (currentIndex < assistantText.length) {
            typingText.textContent = assistantText.slice(0, currentIndex + 1);
            currentIndex++;
        } else {
            clearInterval(animateAssistantText);
            typingCircle.remove();
            const responseSpan = document.createElement('span');
            responseSpan.className = 'response-text';
            typingText.appendChild(responseSpan);
            fetchAndGenerateResponse1(responseSpan, userInput, storeId);
        }
    }, 100);

    // Clear input field
    document.getElementById('user-input').value = '';
}

// Function to fetch and generate AI response
async function fetchAndGenerateResponse1(responseSpan, userInput, storeId) {
    try {
        // Fetch request to the backend
        const response = await fetch('/staff_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: userInput,
                store_id: sessionStorage.getItem('store_id') // Ensure store_id is sent
            }),
        });

        // Parse the response as JSON
        const data = await response.json();
        console.log("Backend Response:", data); // Debugging log

        // Handle backend response
        let finalResponse = "";
        if (data.error) {
            finalResponse = `Error: ${data.error}`;
        } else if (Array.isArray(data.response)) {
            // If the response is an array, format it properly
            finalResponse = data.response.map(row => row[0]).join(", ");
        } else {
            finalResponse = data.response || "No response received.";
        }

        // Animate or display the response
        animateResponse(responseSpan, finalResponse);

    } catch (error) {
        console.error("Error in fetchAndGenerateResponse:", error); // Debugging log
        responseSpan.textContent = "An unexpected error occurred. Please try again later.";
    }
}


// Function to animate the AI response text
function animateResponse(responseSpan, finalResponse) {
    let responseIndex = 0;
    responseSpan.textContent = ''; // Clear previous content

    if (typeof finalResponse !== 'string') {
        finalResponse = JSON.stringify(finalResponse); // Ensure it's a string
    }

    responseInterval = setInterval(() => {
        if (isStopped1) {
            clearInterval(responseInterval);
            return;
        }

        if (responseIndex < finalResponse.length) {
            responseSpan.textContent += finalResponse[responseIndex];
            responseIndex++;

            const chatbox = document.getElementById('chatbox');
            if (chatbox) {
                chatbox.scrollTop = chatbox.scrollHeight;
            }
        } else {
            clearInterval(responseInterval);
            document.getElementById('stop-button').disabled = true;
        }
    }, 50); // Typing speed
}


// Function to stop the AI response animation
function stopResponse() {
    isStopped1 = true;
    clearInterval(responseInterval);
    document.getElementById('stop-button').disabled = true;
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

let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const totalSlides = slides.length;

// Function to show the slide at the specified index
function showSlide(index) {
    currentSlide = (index + totalSlides) % totalSlides; // Wraps around the index
    slides.forEach((slide, i) => {
        slide.style.transition = 'transform 0.5s ease-in-out'; // Add transition effect
        slide.style.transform = `translateX(${100 * (i - currentSlide)}%)`;
    });
}

// Functions to go to the previous or next slide
function prevSlide() {
    showSlide(currentSlide - 1);
}

function nextSlide() {
    showSlide(currentSlide + 1);
}

// Initialize the slider positions on page load
document.addEventListener('DOMContentLoaded', () => {
    slides.forEach((slide, i) => {
        slide.style.transform = `translateX(${100 * i}%)`; // Position slides horizontally
    });
    showSlide(currentSlide);
});

// Dummy stock data for each brand with color and size options
const stockData = {
    nike: {
        colors: ['Red', 'Blue', 'Black', 'White'],
        sizes: ['XS', 'S', 'M', 'L', 'XL'],
        stock: {
            'Red': { 'XS': 10, 'S': 15, 'M': 5, 'L': 0, 'XL': 8 },
            'Blue': { 'XS': 12, 'S': 18, 'M': 7, 'L': 2, 'XL': 10 },
            'Black': { 'XS': 8, 'S': 5, 'M': 14, 'L': 5, 'XL': 3 },
            'White': { 'XS': 0, 'S': 10, 'M': 5, 'L': 10, 'XL': 7 },
        }
    },
    adidas: {
        colors: ['White', 'Black', 'Red', 'Blue'],
        sizes: ['XS', 'S', 'M', 'L', 'XL'],
        stock: {
            'White': { 'XS': 5, 'S': 12, 'M': 8, 'L': 4, 'XL': 0 },
            'Black': { 'XS': 6, 'S': 9, 'M': 14, 'L': 7, 'XL': 10 },
            'Red': { 'XS': 8, 'S': 20, 'M': 6, 'L': 10, 'XL': 3 },
            'Blue': { 'XS': 10, 'S': 5, 'M': 12, 'L': 18, 'XL': 0 },
        }
    },
};


// Update the available colors and sizes based on the selected brand
function updateOptions() {
    const brand = document.getElementById('brand').value;
    const colorSelect = document.getElementById('color');
    const sizeSelect = document.getElementById('size');

    // Clear previous options for color and size
    colorSelect.innerHTML = '';
    sizeSelect.innerHTML = '';

    // Add a default "Select" option for both color and size
    let defaultOption = document.createElement('option');
    defaultOption.textContent = "Select Color";
    defaultOption.value = "";
    colorSelect.appendChild(defaultOption);

    defaultOption = document.createElement('option');
    defaultOption.textContent = "Select Size";
    defaultOption.value = "";
    sizeSelect.appendChild(defaultOption);

    // Populate color options based on the selected brand
    const colors = stockData[brand].colors;
    colors.forEach(color => {
        const option = document.createElement('option');
        option.value = color.toLowerCase(); // Convert to lowercase for consistency
        option.textContent = color;
        colorSelect.appendChild(option);
    });

    // Populate size options based on the selected brand
    const sizes = stockData[brand].sizes;
    sizes.forEach(size => {
        const option = document.createElement('option');
        option.value = size;
        option.textContent = size;
        sizeSelect.appendChild(option);
    });
}

// Check stock for selected brand, color, and size
function checkStock() {
    const brand = document.getElementById('brand').value;
    const color = document.getElementById('color').value;
    const size = document.getElementById('size').value;

    if (color && size) {
        const stock = stockData[brand].stock[color.charAt(0).toUpperCase() + color.slice(1)][size]; // Correct casing for stock data
        const status = stock > 0 ? `Stock available: ${stock}` : 'Out of stock';
        document.getElementById('stock-status').textContent = status;
    } else {
        document.getElementById('stock-status').textContent = 'Please select a color and size.';
    }
}

// Initialize the options for the default brand
window.onload = updateOptions;

// function toggleDropdown() {
//     const dropdown = document.getElementById('dropdown-content');
//     dropdown.classList.toggle('show');
// }

function updateOptions() {
    // Update logic for color or size options (if dynamic behavior is required)
    console.log("Brand selected:", document.getElementById("brand").value);
}

document.addEventListener("DOMContentLoaded", function() {
    fetch('/get_chat_history')
        .then(response => response.json())
        .then(data => {
            const chatHistoryElement = document.getElementById('chat-history');
            chatHistoryElement.innerHTML = ''; // Clear any existing content

            data.forEach(item => {
                const messageElement = document.createElement('div');
                messageElement.classList.add('message');
                messageElement.innerHTML = `
                    <p><strong>User:</strong> ${item.question}</p>
                    <p><strong>AI Assistant:</strong> ${item.response}</p>
                `;
                chatHistoryElement.appendChild(messageElement);
            });
        })
        .catch(error => console.error('Error fetching chat history:', error));
});

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        console.log("File selected:", file.name);
        // Implement the logic to handle file upload here (e.g., send to server)
    }
}

// Server-side chat saving route
app.post('/save-chat', (req, res) => {
    const { userId, title, content } = req.body;
    const query = `INSERT INTO Chats (user_id, title, content) VALUES (?, ?, ?)`;
    db.query(query, [userId, title, content], (err, result) => {
        if (err) return res.status(500).json({ error: err });
        res.json({ message: "Chat saved successfully!" });
    });
});

// Fetch chat history from the server
async function loadChatHistory() {
    const response = await fetch('/get-chats?userId=1'); // Adjust with your endpoint and user ID
    const chats = await response.json();
    const chatList = document.getElementById('chat-sidebar');
    chatList.innerHTML = ''; // Clear any existing chats

    chats.forEach(chat => {
        const chatButton = document.createElement('button');
        chatButton.textContent = chat.title;
        chatButton.onclick = () => openChat(chat.id);
        chatList.appendChild(chatButton);
    });
}

// Function to open a chat when a user clicks on it
function openChat(chatId) {
    fetch(`/get-chat/${chatId}`)
        .then(response => response.json())
        .then(chat => {
            document.getElementById('chat-content').textContent = chat.content;
            document.getElementById('chat-title').textContent = chat.title;
        });
}

// Call loadChatHistory when the page loads
window.onload = loadChatHistory;

// Function to handle sending messages
function sendMessage() {
    let message = document.getElementById("messageInput").value;
    if (message) {
        fetch('/send_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        }).then(response => {
            document.getElementById("messageInput").value = '';
            loadChatHistory();
        });
    }
}

// Function to append messages to chat history
function loadChatHistory() {
    fetch('/get_chat_history')
        .then(response => response.json())
        .then(data => {
            let chatHistoryDiv = document.getElementById("chatHistory");
            chatHistoryDiv.innerHTML = ''; // Clear existing history
            data.forEach(item => {
                let messageElement = document.createElement("div");
                messageElement.className = "chat-message";
                messageElement.textContent = `${item.timestamp}: ${item.message}`;
                chatHistoryDiv.appendChild(messageElement);
            });
        });
}

// Function to handle file uploads
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        appendMessage(`📎 File attached: ${file.name}`, "user");
        console.log("File selected:", file.name); // For debugging
    } else {
        console.log("No file selected");
    }
}

function updateDropdownBackground() {
    const dropdown = document.getElementById('dropdown-content');
    const contentHeight = dropdown.scrollHeight; // Get the full height of the content
    const contentWidth = dropdown.scrollWidth; // Get the full width of the content

    // Dynamically adjust the size or background if needed
    dropdown.style.backgroundSize = `${contentWidth}px ${contentHeight}px`;
}

// Example: Trigger this function whenever dropdown content changes
document.getElementById('dropdown-content').addEventListener('DOMSubtreeModified', updateDropdownBackground);




//attach button


// Handle file input change
function handleFileUpload(event) {
    const file = event.target.files[0];

    if (file) {
        const chatbox = document.getElementById('chatbox');

        const fileMessage = document.createElement('div');
        fileMessage.className = 'chat-message user';
        fileMessage.innerHTML = `<p>📎 ${file.name}</p>`;
        chatbox.appendChild(fileMessage);

        const analyzingMessage = document.createElement('div');
        analyzingMessage.className = 'chat-message gene';
        analyzingMessage.innerHTML = `<p><strong>Analyzing...</strong></p>`;
        chatbox.appendChild(analyzingMessage);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('store_id', 'store2');

        fetch('http://127.0.0.1:5000/analyze_and_summarize', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                    chatbox.removeChild(analyzingMessage);

                    const aiResponse = document.createElement('div');
                    aiResponse.className = 'chat-message ai';

                    if (data.summary && data.responses && data.responses.length > 0) {
                        aiResponse.innerHTML = `
                <p><strong>Summary:</strong> ${data.summary}</p>
                <p><strong>Questions and Responses:</strong></p>
                <ul>
                    ${data.responses.map(resp => `
                        <li>
                            <strong>Q:</strong> ${resp.question.replace(/\s+/g, ' ')}<br>
                            <strong>A:</strong> ${resp.response}
                        </li>
                    `).join('')}
                </ul>
            `;
            } else if (data.error) {
                aiResponse.innerHTML = `<p><strong>Error:</strong> ${data.error}</p>`;
            } else {
                aiResponse.innerHTML = `<p><strong>Questions and Responses:</strong> Unable to extract meaningful Q&A.</p>`;
            }
            chatbox.appendChild(aiResponse);
        })
        .catch(error => {
            chatbox.removeChild(analyzingMessage);
        
            const aiErrorResponse = document.createElement('div');
            aiErrorResponse.className = 'chat-message ai';
            aiErrorResponse.innerHTML = `<p><strong>Error:</strong> ${error.message || 'Failed to process the file. Please check your backend or try again later.'}</p>`;
            chatbox.appendChild(aiErrorResponse);
        });
    }
}






// AI response
// setTimeout(() => {
//     const aiBubble = document.createElement('div');
//     aiBubble.classList.add('chat-bubble', 'ai-bubble');

//     if (userInput.toLowerCase() === 'hi' || userInput.toLowerCase() === 'hii') {
//         aiBubble.innerText = 'Hello! How can I assist you today? 😊';
//     } else {
//         aiBubble.innerText = You said: "${userInput}" - I'm here to help!;
//     }

//     chatbox.appendChild(aiBubble);
//     chatbox.scrollTop = chatbox.scrollHeight;
// }, 500);