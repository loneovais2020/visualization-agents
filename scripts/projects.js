$(document).ready(function () {
    const token = localStorage.getItem("access_token"); // Get token from local storage

    if (!token) {
        window.location.href = "login.html"; // Redirect if not logged in
    }

    // Fetch user projects
    fetch("http://127.0.0.1:8000/projects", {
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
        }
        return response.json();
    })
    .then(data => {
        if (data.detail) {
            $("#projectsContainer").html("<p class='text-center text-danger'>No projects found.</p>");
        } else {
            displayProjects(data);
        }
    })
    .catch(error => console.error("Error fetching projects:", error));

    // Logout function
    $("#logoutBtn").click(function () {
        localStorage.removeItem("access_token");
        window.location.href = "login.html";
    });

    function displayProjects(projects) {
        let projectHTML = "";
        projects.forEach(project => {
            projectHTML += `
                <div class="col-md-4">
                    <div class="card shadow-sm p-3 mb-4 bg-white rounded project-card" data-project-id="${project.project_id}">
                        <div class="card-body">
                            <h5 class="card-title">${project.project_name}</h5>
                            <p class="card-text"><strong>Project ID:</strong> ${project.project_id}</p>
                            <p class="card-text"><strong>Created At:</strong> ${new Date(project.creation_time).toLocaleString()}</p>
                        </div>
                    </div>
                </div>
            `;
        });
        $("#projectsContainer").html(projectHTML);
        
        // Add click event to project cards
        $(".project-card").click(function() {
            const projectId = $(this).data("project-id");
            window.location.href = `project.html?id=${projectId}`;
        });
    }
});
