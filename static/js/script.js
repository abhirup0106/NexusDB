// scripts.js


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
    document.getElementById('contact').style.display = 'none';
}

function showTeam() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'block';
    document.getElementById('about').style.display = 'none';
    document.getElementById('contact').style.display = 'none';
}

function showAbout() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'block';
    document.getElementById('contact').style.display = 'none';
}

function contact() {
    document.getElementById('home').style.display = 'none';
    document.getElementById('Team').style.display = 'none';
    document.getElementById('about').style.display = 'none';
    document.getElementById('contact').style.display = 'block';

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
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userInput })
    });

    const data = await response.json();

    // Display AI assistant's response in the chatbox
    const chatbox = document.getElementById('chatbox');
    const userMessage = `<p><strong>You:</strong> ${userInput}</p>`;
    const aiResponse = `<p><strong>AI Assistant:</strong> ${data.response.result || data.response.error}</p>`;

    chatbox.innerHTML += userMessage + aiResponse;

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
        greetingMessage = "Good Afternoon! Explore the latest T-shirts with AI assistance! ☀️";
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


// let currentSlide = 0;
// const slides = document.querySelectorAll('.slide');
// const totalSlides = slides.length;

// function showSlide(index) {
//     slides.forEach((slide, i) => {
//         slide.style.transform = `translateX(${100 * (i - index)}%)`;
//         slide.style.transition = 'transform 0.5s ease-in-out';
//     });
// }

// function prevSlide() {
//     currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
//     showSlide(currentSlide);
// }

// function nextSlide() {
//     currentSlide = (currentSlide + 1) % totalSlides;
//     showSlide(currentSlide);
// }

// document.addEventListener('DOMContentLoaded', () => {
//     showSlide(currentSlide);
// });


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



//form button
// Dummy stock data for each brand
// Dummy stock data for each brand with color and size options
// Dummy stock data for each brand with color and size options


// Update the available colors and sizes based on the selected brand
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
            'Blue': { 'XS': 10, 'S': 5, 'M': 12, 'L': 18, 'XL': 2 },
        }
    },
    levi: {
        colors: ['Blue', 'Black', 'Gray', 'White'],
        sizes: ['XS', 'S', 'M', 'L', 'XL'],
        stock: {
            'Blue': { 'XS': 12, 'S': 10, 'M': 5, 'L': 3, 'XL': 9 },
            'Black': { 'XS': 15, 'S': 5, 'M': 7, 'L': 8, 'XL': 10 },
            'Gray': { 'XS': 0, 'S': 15, 'M': 10, 'L': 13, 'XL': 12 },
            'White': { 'XS': 5, 'S': 10, 'M': 8, 'L': 0, 'XL': 6 },
        }
    },
    van_huesen: {
        colors: ['Green', 'White', 'Red', 'Black'],
        sizes: ['XS', 'S', 'M', 'L', 'XL'],
        stock: {
            'Green': { 'XS': 8, 'S': 10, 'M': 12, 'L': 5, 'XL': 0 },
            'White': { 'XS': 12, 'S': 6, 'M': 14, 'L': 8, 'XL': 10 },
            'Red': { 'XS': 7, 'S': 18, 'M': 11, 'L': 10, 'XL': 4 },
            'Black': { 'XS': 9, 'S': 15, 'M': 8, 'L': 5, 'XL': 11 },
        }
    }
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


function toggleDropdown() {
    const dropdown = document.getElementById('dropdown-content');
    dropdown.classList.toggle('show');
}

function updateOptions() {
    // Update logic for color or size options (if dynamic behavior is required).
    console.log("Brand selected:", document.getElementById("brand").value);
}

function checkStock() {
    // Example stock-check logic
    const brand = document.getElementById("brand").value;
    const color = document.getElementById("color").value;
    const size = document.getElementById("size").value;

    // Replace this with actual stock-checking logic (e.g., an API call)
    document.getElementById("stock-status").innerText = `Checking stock for ${brand}, ${color}, size ${size}...`;
}


// fetch('/chat_history?user_id=guest')
//     .then(response => {
//         if (!response.ok) {
//             throw new Error('Network response was not ok');
//         }
//         return response.json();
//     })
//     .then(data => {
//         if (data.history) {
//             const historyList = document.getElementById('history-list');
//             data.history.forEach(entry => {
//                 const listItem = document.createElement('li');
//                 listItem.textContent = `${entry.timestamp}: ${entry.question} - ${entry.response}`;
//                 historyList.appendChild(listItem);
//             });
//         } else {
//             console.error('No history data found');
//         }
//     })
//     .catch(error => {
//         console.error('Error fetching history:', error);
//     });

// document.addEventListener('DOMContentLoaded', () => {
//     fetch('/chat_history?user_id=guest')
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error(`HTTP error! Status: ${response.status}`);
//             }
//             return response.json();
//         })
//         .then(data => {
//             console.log('Chat history data:', data); // Debug log
//             const historyList = document.getElementById('history-list');
//             historyList.innerHTML = ''; // Clear existing history

//             if (data.history && data.history.length > 0) {
//                 data.history.forEach(entry => {
//                     const listItem = document.createElement('li');
//                     listItem.textContent = `${entry.timestamp}: ${entry.question} - ${entry.response}`;
//                     historyList.appendChild(listItem);
//                 });
//             } else {
//                 const noHistoryMessage = document.createElement('li');
//                 noHistoryMessage.textContent = 'No chat history available.';
//                 historyList.appendChild(noHistoryMessage);
//             }
//         })
//         .catch(error => {
//             console.error('Error fetching chat history:', error);
//             const historyList = document.getElementById('history-list');
//             historyList.innerHTML = '<li>Error loading chat history. Please try again later.</li>';
//         });
// });




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



app.post('/save-chat', (req, res) => {
    const { userId, title, content } = req.body;
    const query = `INSERT INTO Chats (user_id, title, content) VALUES (?, ?, ?)`;
    db.query(query, [userId, title, content], (err, result) => {
        if (err) return res.status(500).json({ error: err });
        res.json({ message: "Chat saved successfully!" });
    });
});



// Fetch chat history from the server
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












// AI response
// setTimeout(() => {
//     const aiBubble = document.createElement('div');
//     aiBubble.classList.add('chat-bubble', 'ai-bubble');

//     if (userInput.toLowerCase() === 'hi' || userInput.toLowerCase() === 'hii') {
//         aiBubble.innerText = 'Hello! How can I assist you today? 😊';
//     } else {
//         aiBubble.innerText = `You said: "${userInput}" - I'm here to help!`;
//     }

//     chatbox.appendChild(aiBubble);
//     chatbox.scrollTop = chatbox.scrollHeight;
// }, 500);