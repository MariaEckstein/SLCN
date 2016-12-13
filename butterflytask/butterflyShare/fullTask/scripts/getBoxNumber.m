
function deviceNumber = getBoxNumber;

% Checks the connected USB devices and returns the deviceNumber corresponding
% to the button box we use in the scanner. 
% JC 03/02/06

deviceNumber = 0;
d = PsychHID('Devices');
for n = 1:length(d)
    if (d(n).productID == 612) & (strcmp(d(n).usageName,'Keyboard'))
        deviceNumber = n;
    end
end
if deviceNumber == 0
    fprintf(['Button box NOT FOUND.\n']);
end
            