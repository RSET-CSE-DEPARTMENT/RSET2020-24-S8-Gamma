<?php
session_start();

// remove all session variables
session_unset(); 

// destroy the session 
session_destroy(); 

// Redirect to login page
header("Location: login.html");
exit(); // Make sure no code below gets executed after the redirect
?>
