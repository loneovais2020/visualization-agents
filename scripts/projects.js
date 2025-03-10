// Move these functions outside of $(document).ready()
async function deleteProject(projectId, projectName) {
    const token = localStorage.getItem("access_token");
    const BASE_URL = "http://127.0.0.1:8000";

    if (!confirm(`Are you sure you want to delete "${projectName}"? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${BASE_URL}/project/${projectId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            alert("Session expired. Please log in again.");
            localStorage.removeItem("access_token");
            window.location.href = "login.html";
            return;
        }

        if (!response.ok) {
            throw new Error('Failed to delete project');
        }

        // Show success message
        const toastEl = document.getElementById('projectToast');
        const toast = new bootstrap.Toast(toastEl);
        toastEl.querySelector('.toast-body').innerHTML = `
            <div class="text-success">
                <i class="fas fa-check-circle me-2"></i>
                Project "${projectName}" deleted successfully
            </div>
        `;
        toast.show();

        // Refresh projects list
        location.reload();

    } catch (error) {
        console.error('Error deleting project:', error);
        alert('Failed to delete project. Please try again.');
    }
}

async function renameProject(projectId, currentName) {
    const token = localStorage.getItem("access_token");
    const BASE_URL = "http://127.0.0.1:8000";

    const newName = prompt("Enter new project name:", currentName);
    if (!newName || newName === currentName) {
        return;
    }

    try {
        const formData = new FormData();
        formData.append('new_name', newName);

        const response = await fetch(`${BASE_URL}/project/${projectId}/rename`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });

        if (response.status === 401) {
            alert("Session expired. Please log in again.");
            localStorage.removeItem("access_token");
            window.location.href = "login.html";
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to rename project');
        }

        // Show success message
        const toastEl = document.getElementById('projectToast');
        const toast = new bootstrap.Toast(toastEl);
        
        if (result.warning) {
            // Partial success case
            toastEl.querySelector('.toast-body').innerHTML = `
                <div class="text-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${result.warning}
                </div>
            `;
        } else {
            // Complete success case
            toastEl.querySelector('.toast-body').innerHTML = `
                <div class="text-success">
                    <i class="fas fa-check-circle me-2"></i>
                    Project renamed successfully to "${newName}"
                </div>
            `;
        }
        toast.show();

        // Refresh the projects list
        location.reload();

    } catch (error) {
        console.error('Error renaming project:', error);
        
        // Show error message
        const toastEl = document.getElementById('projectToast');
        const toast = new bootstrap.Toast(toastEl);
        toastEl.querySelector('.toast-body').innerHTML = `
            <div class="text-danger">
                <i class="fas fa-times-circle me-2"></i>
                Failed to rename project: ${error.message}
            </div>
        `;
        toast.show();
    }
}

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
                <div class="col-md-4">
                    <div class="card project-card shadow-sm hover-shadow">
                        <div class="project-menu dropdown">
                            <button class="btn btn-light btn-sm rounded-circle" type="button" data-bs-toggle="dropdown">
                                <i class="fas fa-ellipsis-v"></i>
                            </button>
                            <ul class="dropdown-menu dropdown-menu-end">
                                <li>
                                    <a class="dropdown-item" href="#" onclick="renameProject('${project.project_id}', '${project.project_name}')">
                                        <i class="fas fa-edit me-2 text-primary"></i>Rename
                                    </a>
                                </li>
                                <li>
                                    <a class="dropdown-item text-danger" href="#" onclick="deleteProject('${project.project_id}', '${project.project_name}')">
                                        <i class="fas fa-trash-alt me-2"></i>Delete
                                    </a>
                                </li>
                            </ul>
                        </div>
                        <div class="card-body p-4">
                            <h5 class="project-title mb-2">${project.project_name}</h5>
                            <p class="project-date mb-4">
                                <i class="far fa-calendar-alt me-2"></i>${creationDate}
                            </p>
                            <button class="btn btn-primary btn-open w-100" onclick="window.location.href='project.html?id=${project.project_id}'">
                                <i class="fas fa-folder-open me-2"></i>Open Project
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        $("#projectsContainer").html(projectHTML);
    }
});
