def site_mode(request):
    return {
        "site_mode": request.session.get("mode", "recommendation")
    }