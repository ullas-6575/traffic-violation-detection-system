<?php

namespace Tests\Feature;

use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class RoleAccessTest extends TestCase
{
    use RefreshDatabase;

    public function test_guest_is_redirected_from_dashboard(): void
    {
        $this->get('/dashboard')->assertRedirect('/login');
    }

    public function test_officer_can_open_dashboard(): void
    {
        $officer = User::factory()->create(['role' => 'officer']);

        $this->actingAs($officer)->get('/dashboard')->assertOk();
    }

    public function test_admin_can_open_admin_area(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)->get('/admin')->assertOk();
    }

    public function test_officer_cannot_open_admin_area(): void
    {
        $officer = User::factory()->create(['role' => 'officer']);

        $this->actingAs($officer)->get('/admin')->assertForbidden();
    }
}
