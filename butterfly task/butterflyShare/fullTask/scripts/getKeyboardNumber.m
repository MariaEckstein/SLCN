
function k = getKeyboardNumber();

d=PsychHID('Devices');
k = 0;

for n = 1:length(d)
    if strcmp(d(n).usageName,'Keyboard');
        k=n;
        break
    end
end