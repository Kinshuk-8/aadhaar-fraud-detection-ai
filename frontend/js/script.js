// Dynamic API base URL that works in both development and production
const API_BASE_URL = window.location.origin;

// Fallback for development
// const API_BASE_URL = "http://localhost:5000"; // Only for local developmentlet currentMode = "single";

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM loaded - initializing...");
    
    // Hide upload section initially
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    
    // Initialize service cards
    const serviceCards = document.querySelectorAll('.service-card');
    serviceCards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove selected class from all cards
            serviceCards.forEach(c => c.classList.remove('selected'));
            
            // Add selected class to clicked card
            card.classList.add('selected');
            
            // Show upload section
            const uploadSection = document.getElementById('uploadSection');
            if (uploadSection) {
                uploadSection.style.display = 'block';
                // Scroll to upload section for better UX
                uploadSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            // Set mode based on selected service
            const serviceType = card.getAttribute('data-service');
            if (serviceType === 'single' || serviceType === 'batch') {
                setMode(serviceType);
            } else if (serviceType === 'api') {
                // For API service, show a different message
                const uploadTitle = document.getElementById('uploadTitle');
                const uploadDescription = document.getElementById('uploadDescription');
                const verifyForm = document.getElementById('verifyForm');
                
                if (uploadTitle) uploadTitle.textContent = 'API Integration';
                if (uploadDescription) uploadDescription.textContent = 'Contact us to get API access for integrating Aadhaar verification into your applications.';
                if (verifyForm) verifyForm.style.display = 'none';
                
                // Show contact information
                const verificationResults = document.getElementById('verificationResults');
                if (verificationResults) {
                    verificationResults.innerHTML = `
                        <div class="result-card info">
                            <h4>API Integration Service</h4>
                            <p>To integrate our Aadhaar verification service into your applications:</p>
                            <ul style="margin: 15px 0 15px 20px;">
                                <li>Contact our sales team at <strong>sales@aadhaarverify.com</strong></li>
                                <li>Request API documentation and pricing</li>
                                <li>Get your API keys for integration</li>
                                <li>Access our developer portal for implementation guides</li>
                            </ul>
                            <p>We offer flexible pricing plans based on your usage requirements.</p>
                        </div>
                    `;
                }
            }
        });
    });
    
    // Initialize verification page elements
    const singleBtn = document.getElementById("singleBtn");
    const batchBtn = document.getElementById("batchBtn");
    const verifyForm = document.getElementById("verifyForm");
    const closeModal = document.getElementById("closeModal");
    
    if (singleBtn && batchBtn) {
        console.log("Initializing verification page...");
        singleBtn.addEventListener("click", () => setMode("single"));
        batchBtn.addEventListener("click", () => setMode("batch"));
    }
    
    if (verifyForm) {
        verifyForm.addEventListener("submit", handleVerificationSubmit);
        console.log("Form submit listener added");
        
        // Remove required attribute from hidden fields to prevent validation errors
        const zipInput = document.getElementById("zip");
        if (zipInput) {
            zipInput.removeAttribute("required");
        }
    }
    
    if (closeModal) {
        closeModal.addEventListener("click", () => {
            const modal = document.getElementById("resultModal");
            if (modal) modal.style.display = "none";
        });
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', (event) => {
        const modal = document.getElementById("resultModal");
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });
});

function setMode(mode) {
    currentMode = mode;
    const verificationResults = document.getElementById("verificationResults");
    const singleUploadSection = document.getElementById("singleUpload");
    const batchUploadSection = document.getElementById("batchUpload");
    const singleBtn = document.getElementById("singleBtn");
    const batchBtn = document.getElementById("batchBtn");
    const frontInput = document.getElementById("front");
    const zipInput = document.getElementById("zip");
    const qrCheckboxContainer = document.getElementById("qrCheckboxContainer");
    const uploadTitle = document.getElementById("uploadTitle");
    const uploadDescription = document.getElementById("uploadDescription");
    
    console.log(`Setting mode to: ${mode}`);
    
    // Update title and description
    if (uploadTitle && uploadDescription) {
        if (mode === "single") {
            uploadTitle.textContent = "Single Aadhaar Verification";
            uploadDescription.textContent = "Upload Aadhaar card images for verification. Our system automatically detects if the uploaded image is actually an Aadhaar card.";
        } else {
            uploadTitle.textContent = "Batch Aadhaar Verification";
            uploadDescription.textContent = "Upload a ZIP file containing multiple Aadhaar card images for batch processing. Our system will verify each image individually.";
        }
    }
    
    if (verificationResults) verificationResults.innerHTML = "<p>Results will appear here after verification.</p>";
    
    if (mode === "single") {
        if (singleUploadSection) singleUploadSection.style.display = "block";
        if (batchUploadSection) batchUploadSection.style.display = "none";
        if (singleBtn) singleBtn.classList.add("active");
        if (batchBtn) batchBtn.classList.remove("active");
        
        // Show QR checkbox for single mode
        if (qrCheckboxContainer) qrCheckboxContainer.style.display = "block";
        
        // Set required attributes appropriately
        if (frontInput) frontInput.setAttribute("required", "true");
        if (zipInput) zipInput.removeAttribute("required");
        
    } else {
        if (singleUploadSection) singleUploadSection.style.display = "none";
        if (batchUploadSection) batchUploadSection.style.display = "block";
        if (singleBtn) singleBtn.classList.remove("active");
        if (batchBtn) batchBtn.classList.add("active");
        
        // Hide QR checkbox for batch mode
        if (qrCheckboxContainer) qrCheckboxContainer.style.display = "none";
        
        // Set required attributes appropriately
        if (frontInput) frontInput.removeAttribute("required");
        if (zipInput) zipInput.setAttribute("required", "true");
    }
}

function showLoading(show) {
    const loadingSpinner = document.getElementById("loadingSpinner");
    if (loadingSpinner) {
        loadingSpinner.style.display = show ? "block" : "none";
        console.log(`Loading spinner: ${show ? 'shown' : 'hidden'}`);
    }
}

async function handleVerificationSubmit(e) {
    e.preventDefault();
    console.log("Form submitted, mode:", currentMode);
    
    const verificationResults = document.getElementById("verificationResults");
    if (verificationResults) {
        verificationResults.innerHTML = "<p>Processing... Please wait.</p>";
    }
    
    showLoading(true);

    // Create FormData and validate files
    const formData = new FormData();
    
    if (currentMode === "single") {
        const frontFile = document.getElementById("front").files[0];
        
        if (!frontFile) {
            alert("Please select a front image file");
            showLoading(false);
            return;
        }
        
        formData.append("front", frontFile);
        console.log("Front file:", frontFile.name, "size:", frontFile.size);
        
        const backFile = document.getElementById("back").files[0];
        if (backFile) {
            formData.append("back", backFile);
            console.log("Back file:", backFile.name, "size:", backFile.size);
        }
    } else {
        const zipFile = document.getElementById("zip").files[0];
        if (!zipFile) {
            alert("Please select a ZIP file");
            showLoading(false);
            return;
        }
        formData.append("zip", zipFile);
        console.log("ZIP file:", zipFile.name, "size:", zipFile.size);
    }
    
    // Add QR setting (only for single mode)
    if (currentMode === "single") {
        const qrCheckbox = document.querySelector('input[name="qr"]');
        formData.append("qr", qrCheckbox?.checked ? "true" : "false");
        console.log("QR check enabled:", qrCheckbox?.checked);
    }

    const endpoint = currentMode === "single" 
        ? `${API_BASE_URL}/api/verify_single`
        : `${API_BASE_URL}/api/verify_batch`;

    console.log("Sending request to:", endpoint);

    try {
        const response = await fetch(endpoint, { 
            method: "POST", 
            body: formData 
        });
        
        console.log("Response status:", response.status);
        
        if (!response.ok) {
            let errorMessage = `HTTP error! status: ${response.status}`;
            const cloned = response.clone(); // ✅ allows one more read
            try {
                const errorData = await cloned.json();
                errorMessage = errorData.error || errorMessage;
            } catch (e) {
                try {
                    const errorText = await cloned.text();
                    errorMessage = errorText || errorMessage;
                } catch (inner) {
                    console.warn("Could not parse error body:", inner);
                }
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        console.log("API Response received successfully:", data);
        showLoading(false);
        
        if (data.success) {
            if (currentMode === "single") {
                displaySingleResult(data.result);
            } else {
                displayBatchResults(data.results);
            }
        } else {
            // Handle non-Aadhaar case
            if (data.error === 'NOT_AADHAAR') {
                displayNonAadhaarResult(data);
            } else {
                if (verificationResults) {
                    verificationResults.innerHTML = `<p style="color: red;">API Error: ${data.error || "Unknown error"}</p>`;
                }
            }
        }
    } catch (error) {
        console.error("Verification error:", error);
        showLoading(false);
        if (verificationResults) {
            verificationResults.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        }
    }
}

function displaySingleResult(result) {
    const verificationResults = document.getElementById("verificationResults");
    if (!result || !verificationResults) return;

    console.log("Displaying single result:", result);

    // Check if it's a non-Aadhaar result
    if (result.error === 'NOT_AADHAAR') {
        displayNonAadhaarResult(result);
        return;
    }

    const riskLevel = result.assessment || 'UNKNOWN';
    const fraudScore = result.fraud_score || 0;
    
    let riskClass = 'risk-low';
    let riskTagClass = 'risk-low-tag';
    if (riskLevel === 'HIGH') {
        riskClass = 'risk-high';
        riskTagClass = 'risk-high-tag';
    } else if (riskLevel === 'MODERATE') {
        riskClass = 'risk-medium';
        riskTagClass = 'risk-medium-tag';
    }

    // Basic result HTML (only essential info)
    let resultHTML = `
        <div class="result-card success" id="singleResultCard">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4>✅ Valid Aadhaar Card Detected</h4>
                <span class="risk-tag ${riskTagClass}">${riskLevel} RISK</span>
            </div>
            
            <div class="detail-item">
                <span><strong>Fraud Score:</strong></span>
                <span class="${riskClass}">${fraudScore}</span>
            </div>
            
            <div class="detail-item">
                <span><strong>Aadhaar Verification Confidence:</strong></span>
                <span>${result.aadhaar_verification?.confidence_score || 'N/A'}%</span>
            </div>
            
            <div style="margin-top: 20px; text-align: center;">
                <button onclick="showSingleFullDetails(${JSON.stringify(result).replace(/"/g, '&quot;')})" 
                        class="btn" 
                        style="background: #0078d4; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">
                    View Full Details
                </button>
            </div>
        </div>
    `;

    verificationResults.innerHTML = resultHTML;
}

function showSingleFullDetails(result) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    const riskLevel = result.assessment || 'UNKNOWN';
    const fraudScore = result.fraud_score || 0;
    
    let riskClass = 'risk-low';
    let riskTagClass = 'risk-low-tag';
    if (riskLevel === 'HIGH') {
        riskClass = 'risk-high';
        riskTagClass = 'risk-high-tag';
    } else if (riskLevel === 'MODERATE') {
        riskClass = 'risk-medium';
        riskTagClass = 'risk-medium-tag';
    }

    let resultHTML = `
        <div class="result-card success expanded">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4>✅ Valid Aadhaar Card Detected - Full Details</h4>
                <span class="risk-tag ${riskTagClass}">${riskLevel} RISK</span>
            </div>
            
            <div class="detail-item">
                <span><strong>Fraud Score:</strong></span>
                <span class="${riskClass}">${fraudScore}</span>
            </div>
            
            <div class="detail-item">
                <span><strong>Aadhaar Verification Confidence:</strong></span>
                <span>${result.aadhaar_verification?.confidence_score || 'N/A'}%</span>
            </div>
    `;

    // Extracted data
    if (result.extracted) {
        resultHTML += `
            <h5 style="margin-top: 20px; margin-bottom: 10px;">Extracted Information:</h5>
            <div class="verification-details">
                <div class="detail-item">
                    <span><strong>Name:</strong></span>
                    <span>${result.extracted.name || 'Not found'}</span>
                </div>
                <div class="detail-item">
                    <span><strong>Date of Birth:</strong></span>
                    <span>${result.extracted.dob || 'Not found'}</span>
                </div>
                <div class="detail-item">
                    <span><strong>Gender:</strong></span>
                    <span>${result.extracted.gender || 'Not found'}</span>
                </div>
                <div class="detail-item">
                    <span><strong>Aadhaar Number:</strong></span>
                    <span>${result.extracted.aadhaar || 'Not found'}</span>
                </div>
            </div>
        `;
    }

    // Indicators
    if (result.indicators && result.indicators.length > 0) {
        resultHTML += `
            <h5 style="margin-top: 20px; margin-bottom: 10px;">Verification Indicators:</h5>
            <div class="verification-details">
                ${result.indicators.map(indicator => `
                    <div class="detail-item">
                        <span>${indicator}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Processed image
    if (result.annotated_b64) {
        resultHTML += `
            <h5 style="margin-top: 20px; margin-bottom: 10px;">Processed Image:</h5>
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/jpeg;base64,${result.annotated_b64}" 
                     style="max-width: 100%; max-height: 500px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" 
                     alt="Processed Aadhaar image with annotations" />
                <p style="margin-top: 10px; color: #666; font-size: 14px;">Annotated image showing detected elements and verification markers</p>
            </div>
        `;
    }

    // Add collapse button
    resultHTML += `
        <div style="margin-top: 20px; text-align: center;">
            <button onclick="collapseSingleDetails(${JSON.stringify(result).replace(/"/g, '&quot;')})" 
                    class="btn" 
                    style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">
                Show Less
            </button>
        </div>
    `;

    resultHTML += `</div>`;
    verificationResults.innerHTML = resultHTML;

    // Scroll to the results section for better UX
    verificationResults.scrollIntoView({ behavior: 'smooth' });
}

function collapseSingleDetails(result) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    // Go back to basic view
    displaySingleResult(result);
    
    // Scroll to the results section
    verificationResults.scrollIntoView({ behavior: 'smooth' });
}

function displayBatchResults(data) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    const results = data.results || [];
    const summary = data.summary || {};

    if (results.length === 0) {
        verificationResults.innerHTML = `
            <div class="result-card error">
                <h4>❌ No Results</h4>
                <p>No files were processed from the ZIP file.</p>
            </div>
        `;
        return;
    }

    const validResults = results.filter(r => !r.error || r.error === 'NOT_AADHAAR');
    const invalidResults = results.filter(r => r.error === 'NOT_AADHAAR');
    const errorResults = results.filter(r => r.error && r.error !== 'NOT_AADHAAR');

    let resultHTML = `
        <div class="result-card info">
            <h4>📊 Batch Processing Complete</h4>
            
            <!-- Summary Statistics -->
            <div style="margin-bottom: 20px;">
                <h5>Summary</h5>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px;">
                    <div style="text-align: center; padding: 12px; background: #e8f5e8; border-radius: 6px;">
                        <div style="font-size: 20px; font-weight: bold; color: #28a745;">${summary.total_files_processed || results.length}</div>
                        <div style="font-size: 12px;">Total Files</div>
                    </div>
                    <div style="text-align: center; padding: 12px; background: #e8f5e8; border-radius: 6px;">
                        <div style="font-size: 20px; font-weight: bold; color: #28a745;">${summary.valid_aadhaar_cards || validResults.length - invalidResults.length}</div>
                        <div style="font-size: 12px;">Valid Aadhaar</div>
                    </div>
                    <div style="text-align: center; padding: 12px; background: #fff3cd; border-radius: 6px;">
                        <div style="font-size: 20px; font-weight: bold; color: #ffc107;">${summary.non_aadhaar_files || invalidResults.length}</div>
                        <div style="font-size: 12px;">Non-Aadhaar</div>
                    </div>
                    <div style="text-align: center; padding: 12px; background: #f8d7da; border-radius: 6px;">
                        <div style="font-size: 20px; font-weight: bold; color: #dc3545;">${summary.processing_errors || errorResults.length}</div>
                        <div style="font-size: 12px;">Errors</div>
                    </div>
                </div>
                ${summary.success_rate ? `<p style="text-align: center; margin-top: 10px;"><strong>Success Rate:</strong> ${summary.success_rate}</p>` : ''}
            </div>
            
            <!-- Quick File List -->
            <div style="max-height: 300px; overflow-y: auto; margin-bottom: 20px;">
                <h5>File Results</h5>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6;">Filename</th>
                            <th style="padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6;">Status</th>
                            <th style="padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6;">Risk</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    // Show first 10 files in the quick view
    const displayFiles = results.slice(0, 10);
    displayFiles.forEach((result, index) => {
        const isNonAadhaar = result.error === 'NOT_AADHAAR';
        const isError = result.error && result.error !== 'NOT_AADHAAR';
        
        let status = 'Valid Aadhaar';
        let statusColor = '#28a745';
        
        if (isNonAadhaar) {
            status = 'Non-Aadhaar';
            statusColor = '#ffc107';
        } else if (isError) {
            status = 'Error';
            statusColor = '#dc3545';
        }

        resultHTML += `
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                    ${result.filename || `File ${index + 1}`}
                </td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: ${statusColor};">
                    ${status}
                </td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                    ${isNonAadhaar || isError ? 'N/A' : (result.assessment || 'UNKNOWN')}
                </td>
            </tr>
        `;
    });

    // Show message if there are more files
    if (results.length > 10) {
        resultHTML += `
            <tr>
                <td colspan="3" style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6; font-style: italic;">
                    ... and ${results.length - 10} more files
                </td>
            </tr>
        `;
    }

    resultHTML += `
                    </tbody>
                </table>
            </div>
            
            <div style="text-align: center;">
                <button onclick="downloadBatchResults(${JSON.stringify(results).replace(/"/g, '&quot;')})" 
                        class="btn" style="background: #28a745; margin-right: 10px;">
                    📥 Download Full Report
                </button>
                <button onclick="showBatchFullDetails(${JSON.stringify(results).replace(/"/g, '&quot;')})" 
                        class="btn" style="background: #0078d4;">
                    View All Details
                </button>
            </div>
        </div>
    `;

    verificationResults.innerHTML = resultHTML;
}

function showBatchFullDetails(results) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    const validResults = results.filter(r => !r.error || r.error !== 'NOT_AADHAAR');
    const invalidResults = results.filter(r => r.error === 'NOT_AADHAAR');
    const errorResults = results.filter(r => r.error && r.error !== 'NOT_AADHAAR');

    let resultHTML = `
        <div class="result-card info expanded">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h4>📊 Batch Processing - Full Details</h4>
                <button onclick="collapseBatchDetails(${JSON.stringify(results).replace(/"/g, '&quot;')})" 
                        class="btn" 
                        style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Summary
                </button>
            </div>
            
            <!-- Summary Cards -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 30px;">
                <div style="text-align: center; padding: 15px; background: #e8f5e8; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold; color: #28a745;">${validResults.length}</div>
                    <div>Valid Aadhaar</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #fff3cd; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold; color: #ffc107;">${invalidResults.length}</div>
                    <div>Non-Aadhaar</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #f8d7da; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold; color: #dc3545;">${errorResults.length}</div>
                    <div>Errors</div>
                </div>
            </div>
    `;

    // Detailed table for all results
    if (results.length > 0) {
        resultHTML += `
            <h5 style="margin-bottom: 15px;">Detailed Results:</h5>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600;">Filename</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600;">Status</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600;">Risk Level</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600;">Fraud Score</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600;">Aadhaar Number</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        results.forEach((result, index) => {
            const isNonAadhaar = result.error === 'NOT_AADHAAR';
            const isError = result.error && result.error !== 'NOT_AADHAAR';
            const riskLevel = result.assessment || 'UNKNOWN';
            const fraudScore = result.fraud_score || 0;
            const aadhaarNumber = result.extracted?.aadhaar || 'Not found';
            
            // Handle confidence display - ONLY show N/A for non-Aadhaar files
            let confidenceDisplay;
            if (isNonAadhaar) {
                confidenceDisplay = 'N/A'; // Only non-Aadhaar files get N/A
            } else {
                const confidence = result.aadhaar_verification?.confidence_score;
                confidenceDisplay = confidence ? `${confidence}%` : 'N/A';
            }
            
            let status = 'Valid Aadhaar';
            let statusClass = 'status-valid';
            let riskTagClass = 'risk-low-tag';
            
            if (isNonAadhaar) {
                status = 'Non-Aadhaar';
                statusClass = 'status-warning';
                riskTagClass = 'risk-info-tag';
            } else if (isError) {
                status = 'Error';
                statusClass = 'status-error';
                riskTagClass = 'risk-high-tag';
            } else {
                if (riskLevel === 'HIGH') riskTagClass = 'risk-high-tag';
                else if (riskLevel === 'MODERATE') riskTagClass = 'risk-medium-tag';
            }

            const rowClass = index % 2 === 0 ? 'style="background: #f8f9fa;"' : 'style="background: white;"';
            
            resultHTML += `
                <tr ${rowClass}>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        <strong>${result.filename || `File ${index + 1}`}</strong>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        <span class="status-badge ${statusClass}">${status}</span>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        <span class="risk-tag ${riskTagClass}">${isNonAadhaar ? 'N/A' : riskLevel}</span>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        ${isNonAadhaar || isError ? 'N/A' : fraudScore}
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        ${isNonAadhaar || isError ? 'N/A' : aadhaarNumber}
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        ${confidenceDisplay}
                    </td>
                </tr>
            `;
        });

        resultHTML += `
                    </tbody>
                </table>
            </div>
        `;
    }

    resultHTML += `</div>`;
    verificationResults.innerHTML = resultHTML;

    // Scroll to the results section for better UX
    verificationResults.scrollIntoView({ behavior: 'smooth' });
}

function collapseBatchDetails(results) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    // Go back to basic view
    displayBatchResults(results);
    
    // Scroll to the results section
    verificationResults.scrollIntoView({ behavior: 'smooth' });
}

function displayNonAadhaarResult(result) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    const confidence = result.confidence_score || 0;
    const details = result.aadhaar_verification_details || {};
    
    let confidenceColor = '#ff4d4f';
    if (confidence >= 50) confidenceColor = '#ffa500';
    if (confidence >= 70) confidenceColor = '#52c41a';

    verificationResults.innerHTML = `
        <div class="non-aadhaar-alert">
            <div class="non-aadhaar-header">
                <h4>⚠️ Not an Aadhaar Card</h4>
            </div>
            
            <p><strong>Message:</strong> ${result.message || 'The uploaded image does not appear to be an Aadhaar card'}</p>
            
            <div style="margin: 20px 0;">
                <strong>Confidence Score: ${confidence}%</strong>
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: ${confidence}%; background: ${confidenceColor};"></div>
                </div>
            </div>

            <h5>Verification Details:</h5>
            <div class="verification-details">
                <div class="detail-item">
                    <span><strong>Aadhaar Keywords Found:</strong></span>
                    <span class="status-badge ${details.keywords_found >= 2 ? 'status-valid' : 'status-invalid'}">
                        ${details.keywords_found || 0} detected
                    </span>
                </div>
                <div class="detail-item">
                    <span><strong>Aadhaar Number Patterns:</strong></span>
                    <span class="status-badge ${details.aadhaar_numbers_found > 0 ? 'status-valid' : 'status-invalid'}">
                        ${details.aadhaar_numbers_found || 0} found
                    </span>
                </div>
                <div class="detail-item">
                    <span><strong>Image Aspect Ratio:</strong></span>
                    <span class="status-badge ${details.aspect_ratio_valid ? 'status-valid' : 'status-invalid'}">
                        ${details.aspect_ratio_valid ? 'Valid' : 'Invalid'}
                    </span>
                </div>
                <div class="detail-item">
                    <span><strong>Image Size:</strong></span>
                    <span class="status-badge ${details.size_valid ? 'status-valid' : 'status-invalid'}">
                        ${details.size_valid ? 'Adequate' : 'Too Small'}
                    </span>
                </div>
            </div>

            ${details.detected_text_snippets ? `
                <h5>Detected Text Sample:</h5>
                <div class="text-sample">
                    ${details.detected_text_snippets}
                </div>
            ` : ''}

            <div class="result-card info" style="margin-top: 20px;">
                <h5>💡 Recommendation</h5>
                <p>Please upload a clear image of an Aadhaar card that contains:</p>
                <ul style="margin: 10px 0 10px 20px;">
                    <li>Aadhaar-specific text like "Aadhaar", "UIDAI", "Government of India"</li>
                    <li>12-digit Aadhaar number in format XXXX XXXX XXXX</li>
                    <li>Clear photo, name, and demographic information</li>
                    <li>Proper rectangular aspect ratio (similar to a card)</li>
                </ul>
            </div>
        </div>
    `;
}
// Add these functions to your existing script.js file

// Download functionality for single verification
function downloadSingleResult(result) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `aadhaar_verification_${timestamp}`;
    
    // Create comprehensive data structure
    const downloadData = {
        verification_type: "single",
        timestamp: new Date().toISOString(),
        filename: result.filename || "single_verification",
        overall_assessment: result.assessment || "UNKNOWN",
        fraud_score: result.fraud_score || 0,
        is_aadhaar_card: true,
        aadhaar_verification_confidence: result.aadhaar_verification?.confidence_score || "N/A",
        
        // Extracted information
        extracted_data: result.extracted || {},
        
        // OCR data
        ocr_results: result.ocr_data || {},
        
        // QR data if available
        qr_data: result.qr_data || {},
        
        // Verification indicators
        indicators: result.indicators || [],
        
        // Detailed validation results
        validation_results: {
            aadhaar_number: validateAadhaarNumber(result.extracted?.aadhaar),
            name: validateName(result.extracted?.name),
            dob: validateDOB(result.extracted?.dob),
            gender: validateGender(result.extracted?.gender)
        },
        
        // Risk assessment
        risk_breakdown: {
            text_extraction_risk: calculateTextExtractionRisk(result),
            data_consistency_risk: calculateDataConsistencyRisk(result),
            image_quality_risk: calculateImageQualityRisk(result),
            overall_risk_level: result.assessment || "UNKNOWN"
        }
    };
    
    // Offer both JSON and CSV formats
    const jsonData = JSON.stringify(downloadData, null, 2);
    const csvData = convertSingleToCSV(downloadData);
    
    // Create download modal or prompt
    showDownloadOptions(filename, jsonData, csvData);
}

// Download functionality for batch verification
function downloadBatchResults(results) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `batch_aadhaar_verification_${timestamp}`;
    
    // Create comprehensive batch data structure
    const batchData = {
        verification_type: "batch",
        timestamp: new Date().toISOString(),
        total_files: results.length,
        summary: generateBatchSummary(results),
        detailed_results: results.map(result => ({
            filename: result.filename,
            is_aadhaar_card: !result.error || result.error !== 'NOT_AADHAAR',
            error_type: result.error,
            overall_assessment: result.assessment || "UNKNOWN",
            fraud_score: result.fraud_score || 0,
            aadhaar_verification_confidence: result.aadhaar_verification?.confidence_score || "N/A",
            extracted_data: result.extracted || {},
            ocr_results: result.ocr_data || {},
            qr_data: result.qr_data || {},
            indicators: result.indicators || [],
            validation_results: {
                aadhaar_number: validateAadhaarNumber(result.extracted?.aadhaar),
                name: validateName(result.extracted?.name),
                dob: validateDOB(result.extracted?.dob),
                gender: validateGender(result.extracted?.gender)
            }
        }))
    };
    
    const jsonData = JSON.stringify(batchData, null, 2);
    const csvData = convertBatchToCSV(batchData);
    
    showDownloadOptions(filename, jsonData, csvData);
}

// Helper functions for validation (add these to your script)
function validateAadhaarNumber(aadhaar) {
    if (!aadhaar) return { status: "MISSING", valid: false };
    const cleaned = aadhaar.replace(/\s/g, '');
    if (!/^\d{12}$/.test(cleaned)) return { status: "INVALID_FORMAT", valid: false };
    return { status: "VALID", valid: true };
}

function validateName(name) {
    if (!name) return { status: "MISSING", valid: false };
    if (name.length < 2) return { status: "TOO_SHORT", valid: false };
    return { status: "VALID", valid: true };
}

function validateDOB(dob) {
    if (!dob) return { status: "MISSING", valid: false };
    // Basic date validation
    if (!/\d{1,2}\/\d{1,2}\/\d{4}/.test(dob) && !/^\d{4}$/.test(dob)) {
        return { status: "INVALID_FORMAT", valid: false };
    }
    return { status: "VALID", valid: true };
}

function validateGender(gender) {
    if (!gender) return { status: "MISSING", valid: false };
    const normalized = gender.toLowerCase();
    if (!['male', 'female', 'm', 'f'].includes(normalized)) {
        return { status: "INVALID", valid: false };
    }
    return { status: "VALID", valid: true };
}

function calculateTextExtractionRisk(result) {
    const extractedFields = Object.keys(result.extracted || {}).length;
    const expectedFields = 4; // name, dob, gender, aadhaar
    const completeness = extractedFields / expectedFields;
    
    if (completeness >= 0.75) return "LOW";
    if (completeness >= 0.5) return "MEDIUM";
    return "HIGH";
}

function calculateDataConsistencyRisk(result) {
    // Check if extracted data makes sense
    const inconsistencies = [];
    
    if (result.extracted?.dob) {
        const yearMatch = result.extracted.dob.match(/\d{4}/);
        if (yearMatch) {
            const year = parseInt(yearMatch[0]);
            if (year < 1900 || year > new Date().getFullYear()) {
                inconsistencies.push("DOB_OUT_OF_RANGE");
            }
        }
    }
    
    if (result.extracted?.aadhaar) {
        const cleaned = result.extracted.aadhaar.replace(/\s/g, '');
        if (cleaned.length !== 12) {
            inconsistencies.push("AADHAAR_LENGTH_INVALID");
        }
    }
    
    return inconsistencies.length === 0 ? "LOW" : inconsistencies.length === 1 ? "MEDIUM" : "HIGH";
}

function calculateImageQualityRisk(result) {
    // This would typically use image quality metrics from the backend
    // For now, we'll use OCR confidence and presence of face as proxies
    const confidence = result.aadhaar_verification?.confidence_score || 0;
    const hasFace = result.indicators?.some(ind => ind.includes("Face detected"));
    
    if (confidence >= 80 && hasFace) return "LOW";
    if (confidence >= 60) return "MEDIUM";
    return "HIGH";
}

function generateBatchSummary(results) {
    const validAadhaar = results.filter(r => !r.error || r.error !== 'NOT_AADHAAR').length;
    const nonAadhaar = results.filter(r => r.error === 'NOT_AADHAAR').length;
    const errors = results.filter(r => r.error && r.error !== 'NOT_AADHAAR').length;
    
    const riskBreakdown = {
        LOW: results.filter(r => r.assessment === 'LOW').length,
        MODERATE: results.filter(r => r.assessment === 'MODERATE').length,
        HIGH: results.filter(r => r.assessment === 'HIGH').length
    };
    
    return {
        total_processed: results.length,
        valid_aadhaar_cards: validAadhaar,
        non_aadhaar_files: nonAadhaar,
        processing_errors: errors,
        risk_distribution: riskBreakdown,
        overall_risk_score: calculateOverallBatchRisk(results)
    };
}

function calculateOverallBatchRisk(results) {
    if (results.length === 0) return "UNKNOWN";
    
    const riskScores = {
        'LOW': 1,
        'MODERATE': 2,
        'HIGH': 3
    };
    
    const totalRisk = results.reduce((sum, result) => {
        return sum + (riskScores[result.assessment] || 0);
    }, 0);
    
    const averageRisk = totalRisk / results.length;
    
    if (averageRisk >= 2.5) return "HIGH";
    if (averageRisk >= 1.5) return "MODERATE";
    return "LOW";
}

// CSV Conversion functions
function convertSingleToCSV(data) {
    const headers = [
        'Field', 'Value', 'Status', 'Risk Level', 'Details'
    ];
    
    const rows = [
        // Basic info
        ['Verification Type', data.verification_type, 'N/A', 'N/A', ''],
        ['Timestamp', data.timestamp, 'N/A', 'N/A', ''],
        ['Overall Assessment', data.overall_assessment, 'N/A', data.overall_assessment, ''],
        ['Fraud Score', data.fraud_score, 'N/A', 'N/A', ''],
        ['Aadhaar Verification Confidence', data.aadhaar_verification_confidence, 'N/A', 'N/A', ''],
        
        // Extracted data
        ['Aadhaar Number', data.extracted_data.aadhaar || 'N/A', 
         data.validation_results.aadhaar_number.status, 
         data.validation_results.aadhaar_number.valid ? 'LOW' : 'HIGH', ''],
        ['Name', data.extracted_data.name || 'N/A', 
         data.validation_results.name.status,
         data.validation_results.name.valid ? 'LOW' : 'MEDIUM', ''],
        ['Date of Birth', data.extracted_data.dob || 'N/A',
         data.validation_results.dob.status,
         data.validation_results.dob.valid ? 'LOW' : 'MEDIUM', ''],
        ['Gender', data.extracted_data.gender || 'N/A',
         data.validation_results.gender.status,
         data.validation_results.gender.valid ? 'LOW' : 'MEDIUM', ''],
        
        // Risk breakdown
        ['Text Extraction Risk', data.risk_breakdown.text_extraction_risk, 'N/A', data.risk_breakdown.text_extraction_risk, ''],
        ['Data Consistency Risk', data.risk_breakdown.data_consistency_risk, 'N/A', data.risk_breakdown.data_consistency_risk, ''],
        ['Image Quality Risk', data.risk_breakdown.image_quality_risk, 'N/A', data.risk_breakdown.image_quality_risk, '']
    ];
    
    // Add indicators
    data.indicators.forEach((indicator, index) => {
        rows.push([`Indicator ${index + 1}`, indicator, 'N/A', 'N/A', '']);
    });
    
    return [headers, ...rows].map(row => 
        row.map(field => `"${String(field).replace(/"/g, '""')}"`).join(',')
    ).join('\n');
}

function convertBatchToCSV(data) {
    const headers = [
        'Filename', 'Is Aadhaar Card', 'Error Type', 'Overall Assessment', 
        'Fraud Score', 'Confidence Score', 'Aadhaar Number', 'Name', 
        'Date of Birth', 'Gender', 'Aadhaar Validation', 'Name Validation',
        'DOB Validation', 'Gender Validation', 'Risk Level'
    ];
    
    const rows = data.detailed_results.map(result => [
        result.filename,
        result.is_aadhaar_card ? 'Yes' : 'No',
        result.error_type || 'N/A',
        result.overall_assessment,
        result.fraud_score,
        result.aadhaar_verification_confidence,
        result.extracted_data.aadhaar || 'N/A',
        result.extracted_data.name || 'N/A',
        result.extracted_data.dob || 'N/A',
        result.extracted_data.gender || 'N/A',
        result.validation_results.aadhaar_number.status,
        result.validation_results.name.status,
        result.validation_results.dob.status,
        result.validation_results.gender.status,
        result.overall_assessment
    ]);
    
    return [headers, ...rows].map(row => 
        row.map(field => `"${String(field).replace(/"/g, '""')}"`).join(',')
    ).join('\n');
}

// Download modal and file creation
function showDownloadOptions(filename, jsonData, csvData) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
    `;
    
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: #111111;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #222222;
        max-width: 500px;
        width: 90%;
        text-align: center;
    `;
    
    modalContent.innerHTML = `
        <h3 style="color: #e0e0e0; margin-bottom: 20px;">Download Verification Results</h3>
        <p style="color: #aaaaaa; margin-bottom: 25px;">Choose your preferred format:</p>
        <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
            <button onclick="downloadFile('${filename}.json', '${btoa(unescape(encodeURIComponent(jsonData)))}', 'application/json')" 
                    class="btn" style="background: #d40000;">
                Download JSON
            </button>
            <button onclick="downloadFile('${filename}.csv', '${btoa(unescape(encodeURIComponent(csvData)))}', 'text/csv')" 
                    class="btn" style="background: #0078d4;">
                Download CSV
            </button>
            <button onclick="this.closest('[style]').remove()" 
                    class="btn" style="background: #666666;">
                Cancel
            </button>
        </div>
        <p style="color: #888888; margin-top: 20px; font-size: 0.9em;">
            JSON: Complete structured data<br>
            CSV: Spreadsheet-friendly format
        </p>
    `;
    
    modal.appendChild(modalContent);
    document.body.appendChild(modal);
    
    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function downloadFile(filename, base64Data, mimeType) {
    const link = document.createElement('a');
    link.href = `data:${mimeType};base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Close modal after download
    const modal = document.querySelector('div[style*="rgba(0,0,0,0.7)"]');
    if (modal) modal.remove();
}

// Update the display functions to include download buttons

// Add this to displaySingleResult function after the basic result HTML
function addDownloadButtonToSingle(result) {
    return `
        <div style="margin-top: 20px; text-align: center;">
            <button onclick="downloadSingleResult(${JSON.stringify(result).replace(/"/g, '&quot;')})" 
                    class="btn" style="background: #28a745; margin-right: 10px;">
                📥 Download Detailed Report
            </button>
            <button onclick="showSingleFullDetails(${JSON.stringify(result).replace(/"/g, '&quot;')})" 
                    class="btn" style="background: #0078d4;">
                View Full Details
            </button>
        </div>
    `;
}

// Add this to displayBatchResults function
function addDownloadButtonToBatch(results) {
    return `
        <div style="margin-top: 20px; text-align: center;">
            <button onclick="downloadBatchResults(${JSON.stringify(results).replace(/"/g, '&quot;')})" 
                    class="btn" style="background: #28a745; margin-right: 10px;">
                📥 Download Batch Report
            </button>
            <button onclick="showBatchFullDetails(${JSON.stringify(results).replace(/"/g, '&quot;')})" 
                    class="btn" style="background: #0078d4;">
                View Full Details
            </button>
        </div>
    `;
}

// Update the existing display functions to include download buttons
// In displaySingleResult function, add the download button:
function displaySingleResult(result) {
    const verificationResults = document.getElementById("verificationResults");
    if (!result || !verificationResults) return;

    console.log("Displaying single result:", result);

    if (result.error === 'NOT_AADHAAR') {
        displayNonAadhaarResult(result);
        return;
    }

    const riskLevel = result.assessment || 'UNKNOWN';
    const fraudScore = result.fraud_score || 0;
    
    let riskClass = 'risk-low';
    let riskTagClass = 'risk-low-tag';
    if (riskLevel === 'HIGH') {
        riskClass = 'risk-high';
        riskTagClass = 'risk-high-tag';
    } else if (riskLevel === 'MODERATE') {
        riskClass = 'risk-medium';
        riskTagClass = 'risk-medium-tag';
    }

    let resultHTML = `
        <div class="result-card success" id="singleResultCard">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4>✅ Valid Aadhaar Card Detected</h4>
                <span class="risk-tag ${riskTagClass}">${riskLevel} RISK</span>
            </div>
            
            <div class="detail-item">
                <span><strong>Fraud Score:</strong></span>
                <span class="${riskClass}">${fraudScore}</span>
            </div>
            
            <div class="detail-item">
                <span><strong>Aadhaar Verification Confidence:</strong></span>
                <span>${result.aadhaar_verification?.confidence_score || 'N/A'}%</span>
            </div>
            
            ${addDownloadButtonToSingle(result)}
        </div>
    `;

    verificationResults.innerHTML = resultHTML;
}

// Update displayBatchResults function similarly
function displayBatchResults(results) {
    const verificationResults = document.getElementById("verificationResults");
    if (!verificationResults) return;

    if (!results || results.length === 0) {
        verificationResults.innerHTML = `
            <div class="result-card error">
                <h4>❌ No Results</h4>
                <p>No valid results returned from batch processing.</p>
            </div>
        `;
        return;
    }

    const validResults = results.filter(r => !r.error || r.error !== 'NOT_AADHAAR');
    const invalidResults = results.filter(r => r.error === 'NOT_AADHAAR');
    const errorResults = results.filter(r => r.error && r.error !== 'NOT_AADHAAR');

    let resultHTML = `
        <div class="result-card info">
            <h4>📊 Batch Processing Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div style="text-align: center; padding: 15px; background: #e8f5e8; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold; color: #28a745;">${validResults.length}</div>
                    <div>Valid Aadhaar</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #fff3cd; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold; color: #ffc107;">${invalidResults.length}</div>
                    <div>Non-Aadhaar</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #f8d7da; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold; color: #dc3545;">${errorResults.length}</div>
                    <div>Errors</div>
                </div>
            </div>
            
            ${addDownloadButtonToBatch(results)}
        </div>
    `;

    verificationResults.innerHTML = resultHTML;
}

// Make functions globally available
window.downloadSingleResult = downloadSingleResult;
window.downloadBatchResults = downloadBatchResults;
window.downloadFile = downloadFile;
// Make functions globally available
window.showSingleFullDetails = showSingleFullDetails;
window.collapseSingleDetails = collapseSingleDetails;
window.showBatchFullDetails = showBatchFullDetails;
window.collapseBatchDetails = collapseBatchDetails;