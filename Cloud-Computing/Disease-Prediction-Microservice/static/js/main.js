$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        $('#after-submit').hide();
        $('#sub-yes').hide();
        $('#sub-no').hide();
        $('#after-pred').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#after-pred').fadeIn(1000);
                if (data!='Healthy Wheat')
                    $('#result').text(' Disease Detected:  ' + data);
                else
                    $('#result').text(data + ' : No Disease Detected! ');
                $('#result').fadeIn(600);
                console.log('Success!');
            },
        });
    });

    $('#btn-yes').click(function () {
        $.ajax({
            type: 'GET',
            url: '/temp',
            data: null,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('#after-submit').show();
                $('#sub-yes').fadeIn(600);
            },
        });
    });

    $('#btn-no').click(function () {
        $.ajax({
            type: 'GET',
            url: '/temp',
            data: null,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('#after-submit').show();
                $('#sub-no').fadeIn(600);
            },
        });
    });
});
