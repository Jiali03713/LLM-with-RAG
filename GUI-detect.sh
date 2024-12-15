if [ -t 0 ] ; then  
    echo "I'm in a terminal"
else
    zenity --info --title "Hello" --text "I'm being run without a terminal"
fi
