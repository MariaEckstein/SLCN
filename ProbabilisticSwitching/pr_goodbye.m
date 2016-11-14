function pr_goodbye

global exp

if strcmp(exp.data.language, 'English')
    m1 = 'Thank you!';
    m2 = 'You have done a great job!';
    m3 = 'The next task will start soon.';
else
    m1 = 'Vielen Dank!';
    m2 = 'Du hast sehr gut gespielt!';
    m3 = 'Deine nächste Aufgabe wird gleich beginnen.';
end

clearpict(5);

preparestring(m1, 5, 0,  100);
preparestring(m2, 5, 0,   50);
preparestring(m3, 5, 0, -100);

drawpict(5);
wait(3000);