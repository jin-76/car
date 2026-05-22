Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectRoot = fso.GetAbsolutePathName(scriptDir & "\..")
pythonw = projectRoot & "\.venv-gpu\pythonw.exe"
python = projectRoot & "\.venv-gpu\python.exe"
app = scriptDir & "\SII_Traffic_App.py"

If fso.FileExists(pythonw) Then
  shell.Run """" & pythonw & """ """ & app & """", 0, False
ElseIf fso.FileExists(python) Then
  shell.Run """" & python & """ """ & app & """", 0, False
Else
  shell.Run "py -3 """ & app & """", 0, False
End If
