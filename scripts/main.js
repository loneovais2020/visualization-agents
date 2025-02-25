document.addEventListener('DOMContentLoaded', function () {
    const BASE_URL = "http://127.0.0.1:8000"; // FastAPI backend URL

    // Check if user is already logged in
    const token = localStorage.getItem("access_token");
    if (token && window.location.pathname === "/login.html") {
        window.location.href = "projects.html"; // Redirect logged-in users to projects page
    }

    // Handle login form submission
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;

            const loginData = { email, password };

            try {
                const response = await fetch(`${BASE_URL}/login`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(loginData)
                });

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.detail || "Login failed");
                }

                // Store token and redirect
                localStorage.setItem("access_token", result.access_token);
                alert("Login successful! Redirecting to projects...");
                window.location.href = "projects.html"; // Redirect to projects page

            } catch (error) {
                alert(error.message);
            }
        });
    }

    // Handle signup form submission
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const full_name = document.getElementById('fullName').value;
            const phone_number = document.getElementById('phoneNumber').value;
            const email = document.getElementById('signupEmail').value;
            const password = document.getElementById('signupPassword').value;

            const signupData = { full_name, phone_number, email, password };

            try {
                const response = await fetch(`${BASE_URL}/signup`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(signupData)
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.detail || "Signup failed");
                }

                alert("Signup successful! Redirecting to login...");
                window.location.href = "login.html"; // Redirect to login

            } catch (error) {
                alert(error.message);
            }
        });
    }

    // Handle logout
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function () {
            localStorage.removeItem("access_token");
            alert("Logged out successfully!");
            window.location.href = "login.html"; // Redirect to login page
        });
    }
});
