
function k = getInternalKeyboardNumber();

d=PsychHID('Devices');

k = 0;

for n = 1:length(d)
    if strcmp(d(n).usageName,'Keyboard') && strcmp(d(n).product,'Apple Internal Keyboard / Trackpad')
        k=n;
        break
    end
end