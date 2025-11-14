/*
Google Apps Script Web App Code, for visual oscillator replication app
This file shouldn't be deployed (uploaded to server) together with the other files in the replicationapp
It should be deployed in a different way
We just keep this file in Git anyway for version control, at https://github.com/QualiaResearchInstitute/replicationapp-web

This file needs to be deployed as a web app in Google Apps Script
FIRST-TIME SETUP INSTRUCTIONS:
1. Go to script.google.com
2. Create a new project
3. Replace the default code with this code
4. Set up a Google Sheet with appropriate column headers
5. Update the SPREADSHEET_ID constant below with your sheet ID
6. Deploy as web app with execute permissions for "Anyone"
7. Copy the web app URL and use it in your client code

UPDATE INSTRUCTIONS:
1. Edit this script in the Git repository
2. Go to the spreadsheet at docs.google.com
3. menu Extensions -> App Script
4. Copy&paste the code from this file into the window
5. Deploy
6. Git commit & push
*/

const SPREADSHEET_ID = '1kVsiJbZRbAoKST-8h7aviardARbkNKNDjUjumdOfI8I'; // our Google Sheet ID (found in the sheet URL)
const SHEET_NAME = 'Replication App Data'; // Name of the sheet tab

// Column headers for the Google Sheet
const HEADERS = [
  'Timestamp',
  'Drug',
  'Dose',
  'Method',
  'In Altered State',
  'Time After Drug',
  'Visual Match Rating',
  'Notes',
  'URL',
  'All Parameters JSON',
  // 'Photo Index',
  // 'Photo Filename',
  // 'Parameter Count',
  // 'Canvas Size',
  // 'Screen Resolution',
  // 'User Agent',
];

function doPost(e) {
  try {
    // Parse the incoming JSON data
    const data = JSON.parse(e.postData.contents);
    
    // Get or create the spreadsheet
    const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
    let sheet = spreadsheet.getSheetByName(SHEET_NAME);
    
    // Create sheet if it doesn't exist
    if (!sheet) {
      sheet = spreadsheet.insertSheet(SHEET_NAME);
      // Add headers
      sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
      sheet.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold');
    }
    
    // Prepare the row data
    const rowData = [
      new Date(), // Timestamp
      data.formData.drug || '',
      data.formData.dose || '',
      data.formData.method || '',
      data.formData.inAlteredState || '',
      data.formData.timeAfterDrug || '',
      data.formData.visualMatchRating || '',
      data.formData.notes || '',
      data.url || '',
      JSON.stringify(data.parameters) || '', // All parameters as JSON
      // data.parameters.photoIndex || '',
      // data.parameters.currentPhotoFilename || '',
      // data.parameterCount || Object.keys(data.parameters || {}).length,
      // `${data.parameters.canvasWidth || ''}x${data.parameters.canvasHeight || ''}`,
      // data.parameters.screenResolution || '',
      // data.parameters.userAgent || '',
    ];
    
    // Add the data to the sheet
    sheet.appendRow(rowData);
    
    // Return success response
    return ContentService
      .createTextOutput(JSON.stringify({ 
        success: true, 
        message: 'Data submitted successfully',
        rowNumber: sheet.getLastRow()
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    // Return error response
    return ContentService
      .createTextOutput(JSON.stringify({ 
        success: false, 
        message: 'Error: ' + error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  // Handle GET requests (for testing)
  return ContentService
    .createTextOutput('Replication App Data Collector is running. Use POST to submit data.')
    .setMimeType(ContentService.MimeType.TEXT);
}
