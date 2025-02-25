$(document).ready(function () {
    const token = localStorage.getItem("access_token"); // Get token from local storage
    const BASE_URL = "http://127.0.0.1:8000";

    if (!token) {
        window.location.href = "login.html"; // Redirect if not logged in
    }

    // Fetch user projects
    fetch(`${BASE_URL}/projects`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json"
        }
    })
    .then(response => {
        if (response.status === 401) {
            alert("Session expired. Please log in again.");
            localStorage.removeItem("access_token");
            window.location.href = "login.html";
            return;
        }
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(projects => {
        if (Array.isArray(projects) && projects.length > 0) {
            displayProjects(projects);
        } else {
            $("#projectsContainer").html("<p class='text-center text-muted'>No projects found. Create your first project!</p>");
        }
    })
    .catch(error => {
        console.error("Error fetching projects:", error);
        $("#projectsContainer").html("<p class='text-center text-danger'>Error loading projects. Please try again.</p>");
    });

    // Handle project creation
    $("#createProjectBtn").click(async function() {
        const projectName = $("#projectName").val().trim();
        
        if (!projectName) {
            alert("Please enter a project name");
            return;
        }

        try {
            const response = await fetch(`${BASE_URL}/create-project/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    project_name: projectName,
                    user_id: getUserIdFromToken(token)
                })
            });

            if (response.status === 401) {
                alert("Session expired. Please log in again.");
                localStorage.removeItem("access_token");
                window.location.href = "login.html";
                return;
            }

            const result = await response.json();
            
            if (!result.project_id) {
                throw new Error("Project creation failed");
            }
            
            // Close the modal
            $("#createProjectModal").modal('hide');
            
            // Clear the form
            $("#projectName").val('');
            
            // Redirect to the new project
            window.location.href = `project.html?id=${result.project_id}`;

        } catch (error) {
            console.error("Error creating project:", error);
            alert("Failed to create project. Please try again.");
        }
    });

    // Replace getUserEmailFromToken with getUserIdFromToken
    function getUserIdFromToken(token) {
        try {
            const payload = token.split('.')[1];
            const decodedPayload = atob(payload);
            const payloadData = JSON.parse(decodedPayload);
            console.log("Token payload:", payloadData); // Debug line
            return payloadData.user_id;
        } catch (error) {
            console.error("Error decoding token:", error);
            return null;
        }
    }

    // Logout function
    $("#logoutBtn").click(function () {
        localStorage.removeItem("access_token");
        window.location.href = "login.html";
    });

    function displayProjects(projects) {
        let projectHTML = "";
        projects.forEach(project => {
            const creationDate = new Date(project.creation_time).toLocaleString();
            projectHTML += `
                <div class="col-md-4 mb-4">
                    <div class="card h-100 shadow-sm hover-shadow">
                        <div class="card-body">
                            <h5 class="card-title text-primary">${project.project_name}</h5>
                            <p class="card-text">
                                <small class="text-muted">Created: ${creationDate}</small>
                            </p>
                            <div class="d-grid">
                                <button class="btn btn-primary" onclick="window.location.href='project.html?id=${project.project_id}'">
                                    <i class="fas fa-folder-open me-2"></i>Open Project
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        $("#projectsContainer").html(projectHTML);
    }
});
