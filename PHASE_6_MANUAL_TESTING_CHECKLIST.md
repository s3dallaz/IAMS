# Phase 6 - AI Assistant Manual Testing Checklist

**Testing Date**: __________  
**Tester Name**: __________  
**Environment**: Development / Staging / Production

---

## Prerequisites

Before starting tests, ensure:

- [ ] Project is running: `streamlit run app.py`
- [ ] Database contains sample data (students, sessions, attendance records)
- [ ] You know at least 2-3 student names in the database
- [ ] `.env` file exists (can be empty or with `OMNIROUTE_API_KEY` set)

---

## Test 1: Access AI Assistant Page

**Objective**: Verify AI Assistant page loads correctly

**Steps**:
1. Open the application in browser
2. Look at the sidebar navigation
3. Click on "AI Assistant"

**Expected Results**:
- [ ] AI Assistant page loads without errors
- [ ] Chat interface is visible
- [ ] Welcome message: "Hi! I'm your AI Attendance Assistant..."
- [ ] "What can I ask?" expandable section is present
- [ ] Input box for questions is visible
- [ ] "Send" and "Clear Chat" buttons are visible
- [ ] Four quick action buttons are visible at bottom

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 2: Overall Statistics Query

**Objective**: Verify assistant answers system-wide statistics questions

**Steps**:
1. In the AI Assistant page, type: `How many students are registered?`
2. Click "Send" or press Enter
3. Wait for response

**Expected Results**:
- [ ] Loading indicator appears briefly
- [ ] Response appears in chat
- [ ] Response shows actual student count from database
- [ ] Response is properly formatted (not an error message)

**Status**: ⬜ Pass  ⬜ Fail  
**Actual student count shown**: __________  
**Notes**: _______________________________________________

---

## Test 3: Student-Specific Query

**Objective**: Verify assistant retrieves individual student data

**Steps**:
1. Pick a student name from your database (e.g., "Ahmed Hassan")
2. Type: `What is [StudentName]'s attendance percentage?`
3. Click "Send"

**Expected Results**:
- [ ] Response includes student's full name
- [ ] Shows total sessions attended
- [ ] Shows present count
- [ ] Shows late count
- [ ] Shows punctuality rate as percentage
- [ ] All numbers match database reality

**Student name tested**: __________  
**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 4: Recent Sessions Query

**Objective**: Verify assistant shows session information

**Steps**:
1. Type: `Show me recent sessions`
2. Click "Send"

**Expected Results**:
- [ ] Response lists actual sessions from database
- [ ] Each session shows: name, status, present count, late count
- [ ] Session names match database
- [ ] Attendance counts are accurate

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 5: Attendance History Query

**Objective**: Verify assistant shows historical records

**Steps**:
1. Pick a student who has attendance records
2. Type: `Show me [StudentName]'s attendance history`
3. Click "Send"

**Expected Results**:
- [ ] Response includes student name
- [ ] Shows list of attendance records
- [ ] Each record has: session name, date/time, status
- [ ] Uses emoji indicators (✅ for present, ⏰ for late)

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 6: Unknown Student Handling

**Objective**: Verify graceful handling of non-existent students

**Steps**:
1. Type: `What is NonExistentStudent123's attendance?`
2. Click "Send"

**Expected Results**:
- [ ] No crash or error page
- [ ] Response indicates student not found
- [ ] Message is clear and helpful
- [ ] No stack traces or technical errors shown

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 7: Ambiguous Student Query

**Objective**: Verify assistant requests clarification for ambiguous names

**Prerequisites**: 
- Ensure you have at least 2 students with similar names (e.g., "Ahmed Ali" and "Ahmed Hassan")

**Steps**:
1. Type: `What is Ahmed's attendance?` (using just the common first name)
2. Click "Send"

**Expected Results**:
- [ ] Response indicates multiple students found
- [ ] Lists the matching student names
- [ ] Asks user to be more specific
- [ ] No data shown for any specific student (avoiding confusion)

**Status**: ⬜ Pass  ⬜ Fail  ⬜ Skip (only 1 student with that name)  
**Notes**: _______________________________________________

---

## Test 8: Unsupported Question Handling

**Objective**: Verify assistant handles out-of-scope questions gracefully

**Steps**:
1. Type: `What's the weather like today?`
2. Click "Send"

**Expected Results**:
- [ ] No crash or error
- [ ] Response acknowledges the question
- [ ] Provides general information about capabilities
- [ ] Suggests example questions user can ask

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 9: Quick Action Buttons

**Objective**: Verify quick action buttons work correctly

**Steps**:
1. Click the "📊 Overall Stats" button
2. Wait for response
3. Try clicking "📅 Recent Sessions" button
4. Try clicking "👥 Total Students" button
5. Try clicking "✅ Attendance Rate" button

**Expected Results**:
- [ ] Each button triggers a query automatically
- [ ] Query appears in chat as user message
- [ ] Assistant responds appropriately to each query
- [ ] No errors or crashes

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 10: Clear Chat Functionality

**Objective**: Verify chat history can be cleared

**Steps**:
1. Ask 3-4 different questions to build chat history
2. Click "Clear Chat" button
3. Observe the interface

**Expected Results**:
- [ ] All previous messages disappear
- [ ] Welcome message may reappear
- [ ] Chat is ready for new questions
- [ ] No error occurs

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 11: Missing API Key Configuration (Optional)

**Objective**: Verify graceful handling when API key is not configured

**Steps**:
1. Open `.env` file
2. Remove or comment out `OMNIROUTE_API_KEY` line
3. Restart the Streamlit application
4. Navigate to AI Assistant
5. Ask any question

**Expected Results**:
- [ ] Response shows configuration error message
- [ ] Message mentions "OMNIROUTE_API_KEY"
- [ ] Message explains how to fix (set in .env file)
- [ ] No crash or technical error
- [ ] Other pages still work

**Status**: ⬜ Pass  ⬜ Fail  ⬜ Skip (API key required)  
**Notes**: _______________________________________________

---

## Test 12: Case Insensitivity

**Objective**: Verify queries work regardless of capitalization

**Steps**:
1. Type: `what is ahmed's attendance?` (all lowercase)
2. Click "Send"
3. Type: `WHAT IS AHMED'S ATTENDANCE?` (all uppercase)
4. Click "Send"

**Expected Results**:
- [ ] Both queries find the student
- [ ] Both show the same data
- [ ] Case doesn't affect functionality

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 13: Empty Query Handling

**Objective**: Verify handling of empty or whitespace-only input

**Steps**:
1. Leave input box empty
2. Click "Send"
3. Try typing only spaces "   "
4. Click "Send"

**Expected Results**:
- [ ] No error occurs
- [ ] Either nothing happens or helpful message shown
- [ ] Application remains functional

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 14: Phase 4/5 Functionality - Student Registry

**Objective**: Ensure AI Assistant doesn't break existing features

**Steps**:
1. Navigate to "Student Registry" from sidebar
2. Try viewing students
3. Try searching for a student
4. Try adding a new student (optional)

**Expected Results**:
- [ ] Student Registry page loads normally
- [ ] All features work as before
- [ ] No errors or issues

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 15: Phase 4/5 Functionality - Session Management

**Objective**: Ensure session management still works

**Steps**:
1. Navigate to "Session Management"
2. View existing sessions
3. Check session details

**Expected Results**:
- [ ] Session Management page loads
- [ ] Sessions display correctly
- [ ] All features functional

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 16: Phase 4/5 Functionality - Statistics & Reports

**Objective**: Ensure statistics and reports still work

**Steps**:
1. Navigate to "Statistics & Reports"
2. View overall statistics
3. Try generating a report (CSV or PDF)

**Expected Results**:
- [ ] Statistics page loads
- [ ] Data displays correctly
- [ ] Reports can be generated
- [ ] No interference from AI Assistant

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Test 17: Phase 4/5 Functionality - Real-Time Attendance

**Objective**: Ensure attendance features still work

**Steps**:
1. Navigate to "Real-Time Attendance"
2. Check if page loads
3. Verify features are accessible

**Expected Results**:
- [ ] Page loads without errors
- [ ] Camera features available (if applicable)
- [ ] Attendance marking works
- [ ] No issues caused by AI Assistant

**Status**: ⬜ Pass  ⬜ Fail  
**Notes**: _______________________________________________

---

## Additional Observations

### Performance
- [ ] Assistant responds within 2-3 seconds
- [ ] No noticeable lag in UI
- [ ] Page remains responsive during queries

### User Experience
- [ ] Interface is intuitive
- [ ] Error messages are helpful
- [ ] Chat history is easy to read
- [ ] Quick actions are useful

### Issues Found
List any bugs, errors, or unexpected behavior:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

## Overall Test Results

**Total Tests Executed**: ____ / 17  
**Tests Passed**: ____  
**Tests Failed**: ____  
**Tests Skipped**: ____  

**Overall Status**: ⬜ All Pass  ⬜ Minor Issues  ⬜ Major Issues  

---

## Recommendations

Based on testing, recommend:

- [ ] Ready for production
- [ ] Minor fixes needed
- [ ] Major fixes required
- [ ] Additional testing needed

**Additional Comments**:
_______________________________________________
_______________________________________________
_______________________________________________

---

**Testing Completed By**: __________  
**Date**: __________  
**Signature**: __________
