; aiVibe Studio installer (Inno Setup 6). Per-user: no admin rights, no services, nothing that runs in the
; background. Build with installer/build.sh, which passes /DAppVer=<version>.
#ifndef AppVer
  #define AppVer "0.0.0"
#endif

[Setup]
AppId={{49667B2F-0C9F-4881-B0DE-2EC8EB29E6A7}
AppName=aiVibe Studio
AppVersion={#AppVer}
AppVerName=aiVibe Studio {#AppVer}
AppPublisher=carbonPrompt
AppPublisherURL=https://github.com/carbonPrompt/aivibe-studio
AppSupportURL=https://github.com/carbonPrompt/aivibe-studio/issues
AppUpdatesURL=https://github.com/carbonPrompt/aivibe-studio/releases
DefaultDirName={localappdata}\Programs\aiVibe Studio
DefaultGroupName=aiVibe Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
AppMutex=Local\aiVibeStudioSingleInstance
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=aiVibeStudio-Setup-{#AppVer}
SetupIconFile=..\ui\icon.ico
UninstallDisplayIcon={app}\ui\icon.ico
UninstallDisplayName=aiVibe Studio
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[InstallDelete]
; an update replaces the app's code completely, so no stale file from an older version survives
Type: filesandordirs; Name: "{app}\studio"
Type: filesandordirs; Name: "{app}\ui"
Type: filesandordirs; Name: "{app}\examples"

[Files]
Source: "..\app.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\studio\*"; DestDir: "{app}\studio"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs
Source: "..\ui\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs
Source: "..\examples\*"; DestDir: "{app}\examples"; Flags: ignoreversion
Source: "bin\uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.lock.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "python.version"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autodesktop}\aiVibe Studio"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\app.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\ui\icon.ico"
Name: "{group}\aiVibe Studio"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\app.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\ui\icon.ico"
Name: "{group}\Repair aiVibe Studio"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup.ps1"" -Root ""{app}"" -Flavour auto -Force -Pause"; WorkingDir: "{app}"; IconFilename: "{app}\ui\icon.ico"
Name: "{group}\Uninstall aiVibe Studio"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\app.pyw"""; WorkingDir: "{app}"; Description: "Start aiVibe Studio"; Flags: postinstall nowait skipifsilent; Check: SetupSucceeded

[UninstallDelete]
; the environment the installer built (rebuilt on reinstall); user data is asked about separately
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\uvcache"
Type: files; Name: "{app}\torch.txt"

[Code]
var
  Flavour: String;
  SetupOk: Boolean;

function SetupSucceeded(): Boolean;
begin
  Result := SetupOk;
end;

function DriverMajor(): Integer;
var
  Res: TExecOutput;
  Code: Integer;
  S: String;
begin
  Result := -1;
  if ExecAndCaptureOutput('nvidia-smi', '--query-gpu=driver_version --format=csv,noheader', '', SW_HIDE,
                          ewWaitUntilTerminated, Code, Res) and (Code = 0) and (GetArrayLength(Res.StdOut) > 0) then
  begin
    S := Trim(Res.StdOut[0]);
    if Pos('.', S) > 0 then
      S := Copy(S, 1, Pos('.', S) - 1);
    Result := StrToIntDef(S, -1);
  end;
end;

function InitializeSetup(): Boolean;
var
  Major: Integer;
begin
  Result := True;
  Major := DriverMajor();
  if Major >= 580 then
    Flavour := 'cu130'
  else if Major >= 560 then
    Flavour := 'cu126'
  else if Major > 0 then
  begin
    if SuppressibleMsgBox('Your NVIDIA driver (version ' + IntToStr(Major) + ') is too old for aiVibe Studio. ' +
         'Update it from nvidia.com (560 or newer) and run this installer again.' + #13#10#13#10 +
         'Install anyway? Pictures will then be made on the CPU, which is very slow.',
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
      Flavour := 'cpu'
    else
      Result := False;
  end
  else
  begin
    if SuppressibleMsgBox('No NVIDIA graphics card was found. aiVibe Studio needs an NVIDIA RTX card ' +
         '(20-series or newer, 8 GB) for normal speed.' + #13#10#13#10 +
         'Install anyway? Pictures will be made on the CPU, which is very slow.',
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
      Flavour := 'cpu'
    else
      Result := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Page: TOutputMarqueeProgressWizardPage;
  Code: Integer;
  LogFile: String;
begin
  if CurStep <> ssPostInstall then
    exit;
  LogFile := ExpandConstant('{localappdata}\aiVibe Studio\data\logs\setup.log');
  Page := CreateOutputMarqueeProgressPage('Getting Python and the AI libraries',
    'Downloading Python and AI libraries (about 3 to 5 GB). This can take 10 to 30 minutes.');
  Page.Show;
  try
    Page.Animate;
    SetupOk := Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\setup.ps1') +
                    '" -Root "' + ExpandConstant('{app}') + '" -Flavour ' + Flavour, ExpandConstant('{app}'),
                    SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
  finally
    Page.Hide;
  end;
  if not SetupOk then
    SuppressibleMsgBox('Setting up Python and the AI libraries did not finish (often a dropped internet connection).' + #13#10#13#10 +
      'Run "Repair aiVibe Studio" from the Start menu to try again. Details: ' + LogFile,
      mbError, MB_OK, IDOK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  Data := ExpandConstant('{localappdata}\aiVibe Studio');
  if DirExists(Data) and (SuppressibleMsgBox('Also delete your settings, trained subjects and downloaded models?' + #13#10#13#10 +
       'Your pictures are never deleted.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES) then
    DelTree(Data, True, True, True);
end;
