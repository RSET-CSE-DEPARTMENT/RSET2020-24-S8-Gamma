<html>
    <head>
        <title>ARGUS</title>

        <link rel="stylesheet" href="live.css">
</head>

<body>
    <div class="background">
        <div class="argus">Argus</div>

        <?php
        session_start();
        if(isset($_SESSION["username"])) {
            $uname=$_SESSION["username"];

            echo '<div class="welcome">';
            echo "<h1>Welcome $uname</h1>";
            echo '</div>';

            echo '<div class="button-container">';
            echo '<h2>Livestream : </h2>';
            echo '<button onclick="redirectToVideoStream1()"><span class="button_top">Camera 1</span></button>';
            echo '<button onclick="redirectToVideoStream2()"><span class="button_top">Camera 2</span></button>';
            #echo '<h2>TimeStamp : </h2>';
            echo '</div>';

            echo '<div class="button-container2">';
                echo '<a href="timestamp.html" class="cta">';
                    echo '<span>View Timestamp</span>';
                    echo '<svg width="15px" height="10px" viewBox="0 0 13 10">';
                        echo '<path d="M1,5 L11,5"></path>';
                        echo '<polyline points="8 1 12 5 8 9"></polyline>';
                    echo '</svg>';
                echo '</a>';
            echo '</div>';

            echo '<div class="logout">';
            echo "<span class='label'><a href='logout.php'>Logout</a></span>";
            echo '</div>';
            ?>

            <script>
                function redirectToVideoStream1() {
                    window.location.href="http://192.168.161.220:8080/video";
                }
                function redirectToVideoStream2(){
                    window.location.href="https://192.168.63.251:8080/video";
                }

                </script>
                <?php
        } else {
            echo "You have no right to be here";
        }
        ?>
        </div>
    </body>
    </html>
    