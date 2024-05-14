Dim result
Dim message

' Check if a command-line argument is provided
If WScript.Arguments.Count > 0 Then
    ' Retrieve the message from the command-line argument
    message = WScript.Arguments(0)
Else
    ' Default message if no argument is provided
    message = "Click OK to redirect to the desired page."
End If

' Display the message in a pop-up window
result = MsgBox(message, vbOKCancel + vbInformation, "Popup Alert")

If result = vbOK Then
    ' Redirect to the desired page
    Set objShell = CreateObject("WScript.Shell")
    objShell.Run "http://127.0.0.1/login.html"
End If