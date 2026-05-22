#define SourceDir "E:\pywork\x-ii"
#define AppName "ai车辆监控哨兵"
#define AppVersion "1.0.0"

[Setup]
AppId={{C36B6F6B-2E48-4BB3-94E8-7C0C4C4F7901}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=SII
DefaultDirName={localappdata}\Programs\ai车辆监控哨兵
DefaultGroupName={#AppName}
DisableDirPage=no
DisableProgramGroupPage=no
OutputDir={#SourceDir}\installer\output
OutputBaseFilename=SII_Traffic_App_Setup
SetupIconFile={#SourceDir}\sii-exe\app.ico
Compression=lzma2/normal
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=2100000000
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\sii-exe\app.ico

[Languages]
Name: "default"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Dirs]
Name: "{app}\sii-exe\logs"
Name: "{app}\runs\annotations"

; 主界面脚本、图标和运行环境。
[Files]
Source: "{#SourceDir}\sii-exe\SII_Traffic_App.py"; DestDir: "{app}\sii-exe"; Flags: ignoreversion
Source: "{#SourceDir}\sii-exe\app.ico"; DestDir: "{app}\sii-exe"; Flags: ignoreversion

Source: "{#SourceDir}\.venv-gpu\*"; DestDir: "{app}\.venv-gpu"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,*.pyo,*.pdb,include\*,Lib\test\*,Lib\site-packages\*,share\*,conda-meta\*,etc\*,Scripts\pip*.exe,Scripts\wheel*.exe"
Source: "{#SourceDir}\.venv-gpu\Lib\site-packages\cv2\*"; DestDir: "{app}\.venv-gpu\Lib\site-packages\cv2"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\.venv-gpu\Lib\site-packages\PIL\*"; DestDir: "{app}\.venv-gpu\Lib\site-packages\PIL"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\.venv-gpu\Lib\site-packages\numpy\*"; DestDir: "{app}\.venv-gpu\Lib\site-packages\numpy"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,*.pyo,*.pdb,tests\*,testing\tests\*"
Source: "{#SourceDir}\.venv-gpu\Lib\site-packages\numpy.libs\*"; DestDir: "{app}\.venv-gpu\Lib\site-packages\numpy.libs"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\.venv-gpu\Lib\site-packages\pillow.libs\*"; DestDir: "{app}\.venv-gpu\Lib\site-packages\pillow.libs"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; C++ 加速检测程序、模型和 TensorRT 缓存。
; CUDA/cuDNN runtime DLLs are copied next to car_speedup.exe and included by this Release wildcard.
Source: "{#SourceDir}\speedup-c\build-release\Release\*"; DestDir: "{app}\speedup-c\build-release\Release"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\speedup-c\models\*"; DestDir: "{app}\speedup-c\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\speedup-c\trt-cache-fp16\*"; DestDir: "{app}\speedup-c\trt-cache-fp16"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; 随包测试视频固定放到 movie 目录，便于界面按文件名自动匹配矫正脚本。
Source: "{#SourceDir}\movie\Traffic IP Camera video[Gr0HpDM8Ki8].fixed.mp4"; DestDir: "{app}\movie"; DestName: "Traffic IP Camera video[Gr0HpDM8Ki8].mp4"; Flags: ignoreversion
Source: "{#SourceDir}\movie\Car Catches on Fire at Witchita Apartment Parking Lot [oQmCTmGlwgA].mp4"; DestDir: "{app}\movie"; Flags: ignoreversion
Source: "{#SourceDir}\movie\Video Project 10.mp4"; DestDir: "{app}\movie"; Flags: ignoreversion
Source: "{#SourceDir}\movie\Car Accident in Parking Lot Captured by SentriForce [X4qMykwdxNo] 480p.mp4"; DestDir: "{app}\movie"; Flags: ignoreversion
Source: "{#SourceDir}\runs\annotations\*.json"; DestDir: "{app}\runs\annotations"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\.venv-gpu\pythonw.exe"; Parameters: """{app}\sii-exe\SII_Traffic_App.py"""; WorkingDir: "{app}"; IconFilename: "{app}\sii-exe\app.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\.venv-gpu\pythonw.exe"; Parameters: """{app}\sii-exe\SII_Traffic_App.py"""; WorkingDir: "{app}"; IconFilename: "{app}\sii-exe\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\.venv-gpu\pythonw.exe"; Parameters: """{app}\sii-exe\SII_Traffic_App.py"""; WorkingDir: "{app}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
